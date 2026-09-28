#!/usr/bin/env python3
"""Comprehensive test suite for Evaluator-Optimizer Reflection Loop & Streaming Telemetry.

Verifies:
1. Pydantic schema validation (EvaluationGrade, EvaluatorRoute).
2. Graph topology and compilation.
3. First-pass acceptance workflow (0 revisions).
4. Multi-round reflection loop with critique injection and revision count tracking.
5. Circuit breaker escalation on max revisions bound.
6. Immediate circuit breaker triggering on guardrail violation.
7. Streaming telemetry collection, latency tracking, and state diff extraction.
8. ACID SQLite checkpointer state persistence across evaluation rounds.
"""

import os
import sqlite3
import tempfile
import unittest
from typing import Any, Dict

# Set dummy key so imports and schema setups don't fail
os.environ.setdefault("GEMINI_API_KEY", "fake_key_for_testing")

from langgraph.checkpoint.sqlite import SqliteSaver
from evaluator_optimizer_workflow import (
    EvaluationGrade,
    EvaluatorRoute,
    EvaluatorState,
    StreamingTelemetryCollector,
    build_evaluator_optimizer_workflow,
    run_evaluator_optimizer_stream,
    evaluate_route,
)


class TestEvaluatorOptimizerWorkflow(unittest.TestCase):
    """Test suite for LangGraph Evaluator-Optimizer reflection and telemetry."""

    def setUp(self):
        """Set up SQLite memory checkpointer for persistence testing."""
        self.conn = sqlite3.connect(":memory:", check_same_thread=False)
        self.checkpointer = SqliteSaver(self.conn)

    def tearDown(self):
        """Clean up SQLite connection."""
        self.conn.close()

    def test_evaluation_grade_pydantic_validation(self):
        """Verify EvaluationGrade model validates score boundaries and constraints."""
        grade = EvaluationGrade(
            technical_depth=8.5,
            factual_grounding=9.0,
            structure_and_clarity=8.0,
            guardrails_pass=True,
            overall_score=8.5,
            passed=True,
            critique=""
        )
        self.assertEqual(grade.overall_score, 8.5)
        self.assertTrue(grade.passed)
        self.assertTrue(grade.guardrails_pass)

        # Verify validation error on score > 10
        with self.assertRaises(Exception):
            EvaluationGrade(
                technical_depth=12.0,
                factual_grounding=9.0,
                structure_and_clarity=8.0,
                overall_score=9.5,
                passed=True
            )

    def test_evaluator_route_schema(self):
        """Verify EvaluatorRoute Pydantic model validates allowed literals."""
        r1 = EvaluatorRoute(next="accept")
        self.assertEqual(r1.next, "accept")
        r2 = EvaluatorRoute(next="drafter")
        self.assertEqual(r2.next, "drafter")
        r3 = EvaluatorRoute(next="circuit_breaker")
        self.assertEqual(r3.next, "circuit_breaker")

    def test_graph_compilation_and_topology(self):
        """Verify Evaluator-Optimizer graph compiles and contains all necessary nodes."""
        graph = build_evaluator_optimizer_workflow(checkpointer=self.checkpointer)
        self.assertIsNotNone(graph)
        nodes = graph.nodes
        self.assertIn("researcher", nodes)
        self.assertIn("drafter", nodes)
        self.assertIn("evaluator", nodes)
        self.assertIn("accept", nodes)
        self.assertIn("circuit_breaker", nodes)
        self.assertIn("__start__", nodes)

    def test_happy_path_first_pass_acceptance(self):
        """Verify high quality draft passes immediately on round 0 without revision."""
        def mock_perfect_evaluator(draft: str, notes: str, rev: int) -> EvaluationGrade:
            return EvaluationGrade(
                technical_depth=9.5,
                factual_grounding=9.5,
                structure_and_clarity=9.0,
                guardrails_pass=True,
                overall_score=9.35,
                passed=True,
                critique=""
            )

        result = run_evaluator_optimizer_stream(
            topic="High Performance Distributed Systems",
            pass_threshold=8.0,
            max_revisions=2,
            custom_eval_fn=mock_perfect_evaluator,
            verbose=False
        )

        self.assertEqual(result["final_decision"], "ACCEPTED")
        self.assertEqual(result["revision_count"], 0)
        self.assertAlmostEqual(result["final_score"], 9.35, places=2)
        self.assertEqual(len(result["evaluation_history"]), 1)

    def test_reflection_loop_revision_and_pass(self):
        """Verify draft failing round 0 receives critique, optimizes on round 1, and passes."""
        def mock_learning_evaluator(draft: str, notes: str, rev: int) -> EvaluationGrade:
            if rev == 0:
                return EvaluationGrade(
                    technical_depth=6.0,
                    factual_grounding=6.0,
                    structure_and_clarity=6.0,
                    guardrails_pass=True,
                    overall_score=6.0,
                    passed=False,
                    critique="Expand on architecture trade-offs and latency benchmarks."
                )
            else:
                return EvaluationGrade(
                    technical_depth=9.0,
                    factual_grounding=9.0,
                    structure_and_clarity=9.0,
                    guardrails_pass=True,
                    overall_score=9.0,
                    passed=True,
                    critique=""
                )

        result = run_evaluator_optimizer_stream(
            topic="Agent Memory Optimization",
            pass_threshold=8.0,
            max_revisions=2,
            custom_eval_fn=mock_learning_evaluator,
            verbose=False
        )

        self.assertEqual(result["final_decision"], "ACCEPTED")
        self.assertEqual(result["revision_count"], 1)
        self.assertEqual(result["final_score"], 9.0)
        self.assertEqual(len(result["evaluation_history"]), 2)
        self.assertIn("trade-offs", result["evaluation_history"][0]["critique"])

    def test_circuit_breaker_max_revisions_exceeded(self):
        """Verify that persistent quality failure trips circuit breaker at max_revisions."""
        def mock_strict_evaluator(draft: str, notes: str, rev: int) -> EvaluationGrade:
            return EvaluationGrade(
                technical_depth=5.0,
                factual_grounding=5.0,
                structure_and_clarity=5.0,
                guardrails_pass=True,
                overall_score=5.0,
                passed=False,
                critique=f"Insufficient technical rigor in revision {rev}."
            )

        result = run_evaluator_optimizer_stream(
            topic="Quantum Computing Algorithms",
            pass_threshold=8.0,
            max_revisions=2,
            custom_eval_fn=mock_strict_evaluator,
            verbose=False
        )

        self.assertEqual(result["final_decision"], "REJECTED_MAX_REVISIONS")
        self.assertEqual(result["revision_count"], 2)
        self.assertEqual(result["final_score"], 5.0)
        # Should have evaluated 3 times: initial (rev 0), rev 1, and rev 2
        self.assertEqual(len(result["evaluation_history"]), 3)

    def test_circuit_breaker_guardrail_violation(self):
        """Verify that security/safety guardrail breach halts workflow immediately."""
        def mock_unsafe_evaluator(draft: str, notes: str, rev: int) -> EvaluationGrade:
            return EvaluationGrade(
                technical_depth=9.0,
                factual_grounding=9.0,
                structure_and_clarity=9.0,
                guardrails_pass=False,  # Policy breach detected!
                overall_score=9.0,
                passed=False,
                critique="Critical: Sensitive internal infrastructure IP leaked."
            )

        result = run_evaluator_optimizer_stream(
            topic="Confidential System Topology",
            pass_threshold=8.0,
            max_revisions=3,
            custom_eval_fn=mock_unsafe_evaluator,
            verbose=False
        )

        self.assertEqual(result["final_decision"], "REJECTED_GUARDRAIL")
        # Halts immediately on round 0
        self.assertEqual(result["revision_count"], 0)
        self.assertEqual(len(result["evaluation_history"]), 1)

    def test_streaming_telemetry_collector(self):
        """Verify StreamingTelemetryCollector tracks durations, state diffs, and summaries."""
        collector = StreamingTelemetryCollector()
        rec1 = collector.record_step("researcher", {"notes": "abc", "status": "done"}, 15.2)
        self.assertEqual(rec1["step_index"], 1)
        self.assertEqual(rec1["node"], "researcher")
        self.assertEqual(rec1["duration_ms"], 15.2)
        self.assertIn("notes", rec1["state_diff_keys"])

        rec2 = collector.record_step("drafter", {"draft": "hello world"}, 32.5)
        self.assertEqual(rec2["step_index"], 2)
        self.assertEqual(rec2["node"], "drafter")
        self.assertIn("draft", rec2["state_diff_keys"])

        summary = collector.get_summary()
        self.assertEqual(summary["total_steps"], 2)
        self.assertIn("researcher", summary["node_durations_ms"])
        self.assertIn("drafter", summary["node_durations_ms"])
        self.assertEqual(summary["node_durations_ms"]["researcher"], 15.2)

    def test_sqlite_checkpointer_integration(self):
        """Verify Evaluator-Optimizer state and thread isolation persist in SQLite."""
        temp_fd, temp_path = tempfile.mkstemp(suffix=".db")
        try:
            conn = sqlite3.connect(temp_path, check_same_thread=False)
            saver = SqliteSaver(conn)

            result1 = run_evaluator_optimizer_stream(
                topic="State Checkpointing Architecture",
                pass_threshold=8.0,
                max_revisions=1,
                checkpointer=saver,
                thread_id="thread-test-checkpoint-1",
                verbose=False
            )
            self.assertIn(result1["final_decision"], ["ACCEPTED", "REJECTED_MAX_REVISIONS"])

            # Verify that checkpoints table exists and has rows
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM checkpoints WHERE thread_id = 'thread-test-checkpoint-1'")
            count = cursor.fetchone()[0]
            self.assertGreater(count, 0)
            conn.close()
        finally:
            os.close(temp_fd)
            if os.path.exists(temp_path):
                os.remove(temp_path)


if __name__ == "__main__":
    unittest.main()
