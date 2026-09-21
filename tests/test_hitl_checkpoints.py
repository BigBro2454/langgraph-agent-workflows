import os
import sqlite3
import tempfile
import unittest
from typing import Annotated, List, Optional, TypedDict
from pydantic import BaseModel

# Set dummy key so imports and schema setups don't fail
os.environ.setdefault("GEMINI_API_KEY", "fake_key_for_testing")

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

import hitl_checkpoint_workflow as hitl
import main


class TestHITLAndSqliteCheckpointing(unittest.TestCase):
    """Test suite for Human-in-the-Loop Breakpoints and SQLite State Checkpointing."""

    def setUp(self):
        """Set up in-memory and temp SQLite environments."""
        self.temp_db_fd, self.temp_db_path = tempfile.mkstemp(suffix=".db")
        self.conn = sqlite3.connect(":memory:", check_same_thread=False)
        self.checkpointer = SqliteSaver(self.conn)

    def tearDown(self):
        """Clean up SQLite resources."""
        self.conn.close()
        os.close(self.temp_db_fd)
        if os.path.exists(self.temp_db_path):
            os.remove(self.temp_db_path)

    def test_sqlite_saver_initialization(self):
        """Verify SqliteSaver compiles properly and creates checkpoint schema."""
        self.assertIsInstance(self.checkpointer, SqliteSaver)
        # Check that sqlite connection is alive
        cursor = self.conn.cursor()
        cursor.execute("SELECT 1")
        self.assertEqual(cursor.fetchone()[0], 1)

    def test_hitl_graph_topology_and_compilation(self):
        """Verify HITL supervisor graph compiles with checkpointer and breakpoints."""
        graph = hitl.build_hitl_workflow(
            checkpointer=self.checkpointer,
            interrupt_before=["writing_team"]
        )
        self.assertIsNotNone(graph)
        nodes = graph.nodes
        self.assertIn("supervisor", nodes)
        self.assertIn("research_team", nodes)
        self.assertIn("writing_team", nodes)
        self.assertIn("__start__", nodes)

    def test_hitl_breakpoint_interruption_and_state_inspection(self):
        """Verify graph halts at specified breakpoint and captures state in SQLite."""
        class MockState(TypedDict):
            messages: Annotated[List[BaseMessage], add_messages]
            step: str

        def research_node(state: MockState):
            return {
                "messages": [AIMessage(content="Market research findings on AI agents.", name="researcher")],
                "step": "researched"
            }

        def writing_node(state: MockState):
            return {
                "messages": [AIMessage(content="Final executive white paper on AI agents.", name="writer")],
                "step": "written"
            }

        builder = StateGraph(MockState)
        builder.add_node("research", research_node)
        builder.add_node("writing", writing_node)
        builder.add_edge(START, "research")
        builder.add_edge("research", "writing")
        builder.add_edge("writing", END)

        # Compile with SqliteSaver and interrupt before 'writing'
        graph = builder.compile(
            checkpointer=self.checkpointer,
            interrupt_before=["writing"]
        )

        config = {"configurable": {"thread_id": "thread-audit-1"}}
        initial_input = {"messages": [HumanMessage(content="Conduct research and draft paper.")]}

        # Step 1: Run until interrupt
        events = list(graph.stream(initial_input, config))
        self.assertTrue(len(events) > 0)

        # Step 2: Inspect paused state from SQLite checkpoint
        state = graph.get_state(config)
        self.assertEqual(state.next, ("writing",))
        self.assertEqual(state.values["step"], "researched")
        self.assertEqual(len(state.values["messages"]), 2)
        self.assertEqual(state.values["messages"][-1].name, "researcher")

        # Step 3: Inject Human Guidance / Feedback before resuming
        human_edit = HumanMessage(
            content="Include specific benchmarks comparing latency and token costs.",
            name="human_supervisor"
        )
        graph.update_state(config, {"messages": [human_edit]})

        # Verify state updated with human feedback
        updated_state = graph.get_state(config)
        self.assertEqual(len(updated_state.values["messages"]), 3)
        self.assertEqual(updated_state.values["messages"][-1].content, human_edit.content)
        self.assertEqual(updated_state.next, ("writing",))

        # Step 4: Resume execution with None
        resume_events = list(graph.stream(None, config))
        self.assertTrue(len(resume_events) > 0)

        # Step 5: Verify completion
        final_state = graph.get_state(config)
        self.assertEqual(final_state.next, ())
        self.assertEqual(final_state.values["step"], "written")
        self.assertEqual(len(final_state.values["messages"]), 4)
        self.assertEqual(final_state.values["messages"][-1].name, "writer")

    def test_checkpoint_history_time_travel(self):
        """Verify SQLite checkpoints record history that can be retrieved for audit trails."""
        class StepState(TypedDict):
            counter: int

        def inc_a(s: StepState):
            return {"counter": s["counter"] + 1}

        def inc_b(s: StepState):
            return {"counter": s["counter"] + 10}

        builder = StateGraph(StepState)
        builder.add_node("step_a", inc_a)
        builder.add_node("step_b", inc_b)
        builder.add_edge(START, "step_a")
        builder.add_edge("step_a", "step_b")
        builder.add_edge("step_b", END)

        graph = builder.compile(checkpointer=self.checkpointer)
        config = {"configurable": {"thread_id": "thread-time-travel"}}

        list(graph.stream({"counter": 0}, config))

        history = list(graph.get_state_history(config))
        # Expect at least initial, step_a, step_b checkpoints
        self.assertGreaterEqual(len(history), 3)

        # Checkpoints are in reverse chronological order
        final_cp = history[0]
        self.assertEqual(final_cp.values["counter"], 11)
        prev_cp = history[1]
        self.assertEqual(prev_cp.values["counter"], 1)

    def test_multi_thread_state_isolation(self):
        """Verify separate thread_ids maintain completely isolated states in SQLite."""
        class ValueState(TypedDict):
            tag: str

        def set_node(s: ValueState):
            return {"tag": s["tag"].upper()}

        builder = StateGraph(ValueState)
        builder.add_node("set_node", set_node)
        builder.add_edge(START, "set_node")
        builder.add_edge("set_node", END)

        graph = builder.compile(checkpointer=self.checkpointer)

        cfg_alpha = {"configurable": {"thread_id": "session-alpha"}}
        cfg_beta = {"configurable": {"thread_id": "session-beta"}}

        list(graph.stream({"tag": "alpha_workflow"}, cfg_alpha))
        list(graph.stream({"tag": "beta_workflow"}, cfg_beta))

        state_alpha = graph.get_state(cfg_alpha)
        state_beta = graph.get_state(cfg_beta)

        self.assertEqual(state_alpha.values["tag"], "ALPHA_WORKFLOW")
        self.assertEqual(state_beta.values["tag"], "BETA_WORKFLOW")
        self.assertNotEqual(state_alpha.values["tag"], state_beta.values["tag"])

    def test_hitl_workflow_manager_api(self):
        """Verify HITLWorkflowManager handles lifecycle, inspection, and feedback."""
        manager = hitl.HITLWorkflowManager(db_path=self.temp_db_path)
        try:
            thread_id = "test-mgr-thread"
            config = manager.get_config(thread_id)
            self.assertEqual(config["configurable"]["thread_id"], thread_id)

            # Inspect state before run
            info = manager.inspect_state(thread_id)
            self.assertFalse(info["is_paused"])
            self.assertEqual(info["thread_id"], thread_id)

            # Feedback injection test
            manager.inject_feedback(thread_id, "Prioritize safety and cost control.")
            inspected = manager.inspect_state(thread_id)
            self.assertIn("Prioritize safety", inspected["messages"][-1]["content_preview"])
        finally:
            manager.close()

    def test_main_compile_graph_with_sqlite(self):
        """Verify foundational main.py chatbot supports SqliteSaver compilation."""
        persistent_graph = main.compile_graph(checkpointer=self.checkpointer)
        self.assertIsNotNone(persistent_graph)
        self.assertIn("chatbot", persistent_graph.nodes)


if __name__ == "__main__":
    unittest.main()
