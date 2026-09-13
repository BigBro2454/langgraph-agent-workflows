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
        description="LangGraph Agent Workflows - Multi-Agent Systems & Cyclic State Machines"
    )
    parser.add_argument(
        "--mode",
        choices=["simple", "hierarchical"],
        default="hierarchical",
        help="Workflow execution mode (default: hierarchical)"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Write an executive briefing on how enterprise multi-agent workflows reduce token latency and prevent context poisoning.",
        help="Prompt for the hierarchical supervisor workflow"
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
        # main.py __name__ == '__main__' block can be invoked directly:
        os.system(f"{sys.executable} main.py")
    else:
        import hierarchical_blog_writer as hbw
        print(f"Starting Hierarchical Multi-Agent Supervisor with Prompt:\n'{args.prompt}'\n")
        os.system(f"{sys.executable} hierarchical_blog_writer.py \"{args.prompt}\"")


if __name__ == "__main__":
    main()
