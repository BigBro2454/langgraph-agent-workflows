#!/usr/bin/env python3
"""Unified CLI Runner for LangGraph Agent Workflows.

Demonstrates:
  1. Foundational cyclic StateGraph agent (interactive chatbot)
  2. Hierarchical multi-agent supervisor workflow (subgraphs as nodes)
"""

import argparse
import os
import sys
import unittest

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def run_tests():
    """Run the test suite and report results."""
    print("🧪 Running LangGraph Workflows Test Suite...\n")
    loader = unittest.TestLoader()
    suite = loader.discover("tests", pattern="test_*.py")
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)


def main():
    parser = argparse.ArgumentParser(
        description="LangGraph Agent Workflows - Multi-Agent Systems, HITL Breakpoints & SQLite Checkpointing"
    )
    parser.add_argument(
        "--mode",
        choices=["simple", "hierarchical", "hitl", "evaluator"],
        default="hierarchical",
        help="Workflow execution mode (default: hierarchical)"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Write an executive briefing on how enterprise multi-agent workflows reduce token latency and prevent context poisoning.",
        help="Prompt or topic for the workflow"
    )
    parser.add_argument(
        "--thread-id",
        type=str,
        default="session-thread-1",
        help="Thread ID for SQLite state checkpointing (used in hitl and evaluator modes)"
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default="checkpoints.db",
        help="SQLite database path for checkpoints"
    )
    parser.add_argument(
        "--pass-threshold",
        type=float,
        default=8.0,
        help="Minimum quality score threshold (0-10) to pass acceptance gate in evaluator mode (default: 8.0)"
    )
    parser.add_argument(
        "--max-revisions",
        type=int,
        default=2,
        help="Maximum self-correction reflection iterations before circuit breaker trips (default: 2)"
    )
    parser.add_argument(
        "--export-telemetry",
        type=str,
        default="",
        help="File path to export JSON streaming telemetry report"
    )
    parser.add_argument(
        "--inspect",
        action="store_true",
        help="Inspect state at breakpoint for thread-id"
    )
    parser.add_argument(
        "--feedback",
        type=str,
        help="Human supervisor feedback/steering to inject into state"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume execution of a paused breakpoint thread"
    )
    parser.add_argument(
        "--history",
        action="store_true",
        help="Print audit history of saved checkpoints for thread-id"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run unit test suite"
    )

    args = parser.parse_args()

    if args.test:
        run_tests()

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("⚠️  WARNING: GEMINI_API_KEY is not set.")
        print("   Set it via: export GEMINI_API_KEY='your-key-here'")
        print("   Or create a .env file from .env.example\n")

    if args.mode == "simple":
        import main as simple_agent
        # Runs the interactive loop in main.py
        print("Starting Simple LangGraph Chatbot Loop...")
        os.system(f"{sys.executable} main.py")
    elif args.mode == "evaluator":
        import json
        import sqlite3
        from langgraph.checkpoint.sqlite import SqliteSaver
        from evaluator_optimizer_workflow import run_evaluator_optimizer_stream

        conn = sqlite3.connect(args.db_path, check_same_thread=False)
        saver = SqliteSaver(conn)
        try:
            result = run_evaluator_optimizer_stream(
                topic=args.prompt,
                pass_threshold=args.pass_threshold,
                max_revisions=args.max_revisions,
                checkpointer=saver,
                thread_id=args.thread_id,
                verbose=True
            )
            if args.export_telemetry:
                telemetry = result.get("telemetry", {})
                with open(args.export_telemetry, "w", encoding="utf-8") as f:
                    json.dump(telemetry, f, indent=2)
                print(f"📊 Telemetry exported cleanly to: {args.export_telemetry}")
        finally:
            conn.close()
    elif args.mode == "hitl":
        from hitl_checkpoint_workflow import HITLWorkflowManager
        manager = HITLWorkflowManager(db_path=args.db_path)
        try:
            if args.inspect:
                info = manager.inspect_state(args.thread_id)
                print(f"\n🔍 State Inspection for Thread '{args.thread_id}':")
                print(f"  - Paused: {info['is_paused']}")
                print(f"  - Next Node: {info['next_node']}")
                print(f"  - Checkpoint ID: {info['checkpoint_id']}")
                print(f"  - Messages ({len(info['messages'])}):")
                for m in info["messages"][-5:]:
                    print(f"    [{m['sender']}]: {m['content_preview']}...")
                return
            if args.history:
                hist = manager.get_history(args.thread_id)
                print(f"\n📜 Checkpoint History ({len(hist)} entries):")
                for idx, h in enumerate(hist):
                    print(f"  #{idx+1} [ID: {h['checkpoint_id'][:8]}...] Next: {h['next']} | Msgs: {h['messages_count']}")
                return
            if args.resume:
                print(f"\n▶️ Resuming thread '{args.thread_id}' with feedback: {args.feedback}...")
                res = manager.resume_workflow(args.thread_id, feedback=args.feedback)
                print(f"Status: {res['status']}, Next: {res['next']}, Total Msgs: {res['messages_count']}")
                return

            print(f"Starting HITL Checkpointed Supervisor (Thread: {args.thread_id})...")
            res = manager.start_workflow(args.thread_id, args.prompt)
            print(f"Status: {res['status']}, Next: {res['next']}, Checkpoint ID: {res['checkpoint_id']}")
        finally:
            manager.close()
    else:
        import hierarchical_blog_writer as hbw
        print(f"Starting Hierarchical Multi-Agent Supervisor with Prompt:\n'{args.prompt}'\n")
        os.system(f"{sys.executable} hierarchical_blog_writer.py \"{args.prompt}\"")


if __name__ == "__main__":
    main()
