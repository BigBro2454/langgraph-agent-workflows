#!/usr/bin/env python3
"""Production Human-in-the-Loop (HITL) Breakpoints & SQLite State Checkpointing.

Demonstrates Google L5 systems architecture for LangGraph workflows:
1. Durable State Persistence with SqliteSaver (thread isolation, crash recovery).
2. Deterministic Human-in-the-Loop Breakpoints (interrupt_before).
3. State Inspection & Runtime Interventions (graph.get_state, graph.update_state).
4. Resumption & Time-Travel Replay (resuming from pause, audit trails).
"""

import os
import sys
import sqlite3
from typing import Annotated, Any, Dict, List, Literal, Optional, Tuple, TypedDict
from pydantic import BaseModel

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# LLM Configuration
api_key = os.environ.get("GEMINI_API_KEY")
model_name = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

llm = ChatGoogleGenerativeAI(
    model=model_name,
    google_api_key=api_key or "placeholder_key"
)


# --- 1. Global State Definition ---
class WorkflowState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    next: str
    approved: Optional[bool]
    human_feedback: Optional[str]


# --- 2. Pydantic Structured Output Routing Models ---
class ResearchRoute(BaseModel):
    next: Literal["search_agent", "web_scraper_agent", "FINISH"]


class WritingRoute(BaseModel):
    next: Literal["doc_writer_agent", "note_taker_agent", "chart_generator_agent", "FINISH"]


class SuperRoute(BaseModel):
    next: Literal["research_team", "writing_team", "FINISH"]


# --- 3. Subgraph Builders ---
def build_research_subgraph() -> Any:
    """Compile the Research Team subgraph."""
    research_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are the supervisor of the Research Team. Your goal is to gather information. "
                   "Your team members are: search_agent, web_scraper_agent. "
                   "When you have gathered enough information, respond with FINISH."),
        MessagesPlaceholder(variable_name="messages"),
        ("system", "Given the conversation above, who should act next? Select one of: search_agent, web_scraper_agent, FINISH")
    ])
    research_supervisor = research_prompt | llm.with_structured_output(ResearchRoute)

    def search_agent_node(state: WorkflowState):
        response = llm.invoke([SystemMessage(content="You are a search agent. Find relevant information on the topic.")] + state["messages"])
        return {"messages": [HumanMessage(content=response.content, name="search_agent")]}

    def web_scraper_agent_node(state: WorkflowState):
        response = llm.invoke([SystemMessage(content="You are a web scraper agent. Extract detailed data to support the research.")] + state["messages"])
        return {"messages": [HumanMessage(content=response.content, name="web_scraper_agent")]}

    research_builder = StateGraph(WorkflowState)
    research_builder.add_node("supervisor", research_supervisor)
    research_builder.add_node("search_agent", search_agent_node)
    research_builder.add_node("web_scraper_agent", web_scraper_agent_node)

    research_builder.add_edge("search_agent", "supervisor")
    research_builder.add_edge("web_scraper_agent", "supervisor")
    research_builder.add_conditional_edges(
        "supervisor",
        lambda x: x["next"],
        {"FINISH": END, "search_agent": "search_agent", "web_scraper_agent": "web_scraper_agent"}
    )
    research_builder.add_edge(START, "supervisor")
    return research_builder.compile()


def build_writing_subgraph() -> Any:
    """Compile the Writing Team subgraph."""
    writing_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are the supervisor of the Writing Team. Your goal is to compile a report based on the research. "
                   "Your team members are: doc_writer_agent, note_taker_agent, chart_generator_agent. "
                   "When the blog post is complete, respond with FINISH."),
        MessagesPlaceholder(variable_name="messages"),
        ("system", "Given the conversation above, who should act next? Select one of: doc_writer_agent, note_taker_agent, chart_generator_agent, FINISH")
    ])
    writing_supervisor = writing_prompt | llm.with_structured_output(WritingRoute)

    def doc_writer_agent_node(state: WorkflowState):
        response = llm.invoke([SystemMessage(content="You are a doc writer agent. Write the final blog post incorporating research notes.")] + state["messages"])
        return {"messages": [HumanMessage(content=response.content, name="doc_writer_agent")]}

    def note_taker_agent_node(state: WorkflowState):
        response = llm.invoke([SystemMessage(content="You are a note taker agent. Organize the research findings into coherent bullet points.")] + state["messages"])
        return {"messages": [HumanMessage(content=response.content, name="note_taker_agent")]}

    def chart_generator_agent_node(state: WorkflowState):
        response = llm.invoke([SystemMessage(content="You are a chart generating agent. Create markdown tables representing key data points.")] + state["messages"])
        return {"messages": [HumanMessage(content=response.content, name="chart_generator_agent")]}

    writing_builder = StateGraph(WorkflowState)
    writing_builder.add_node("supervisor", writing_supervisor)
    writing_builder.add_node("doc_writer_agent", doc_writer_agent_node)
    writing_builder.add_node("note_taker_agent", note_taker_agent_node)
    writing_builder.add_node("chart_generator_agent", chart_generator_agent_node)

    for member in ["doc_writer_agent", "note_taker_agent", "chart_generator_agent"]:
        writing_builder.add_edge(member, "supervisor")

    writing_builder.add_conditional_edges(
        "supervisor",
        lambda x: x["next"],
        {
            "FINISH": END,
            "doc_writer_agent": "doc_writer_agent",
            "note_taker_agent": "note_taker_agent",
            "chart_generator_agent": "chart_generator_agent"
        }
    )
    writing_builder.add_edge(START, "supervisor")
    return writing_builder.compile()


# --- 4. Hierarchical Supervisor Workflow with HITL & SqliteSaver ---
def build_hitl_workflow(
    checkpointer: Optional[SqliteSaver] = None,
    interrupt_before: Optional[List[str]] = None
) -> Any:
    """Build and compile the top-level supervisor graph with checkpoints & breakpoints.
    
    Args:
        checkpointer: SqliteSaver instance for persisting thread states.
        interrupt_before: List of node names before which execution pauses for human approval.
                         Defaults to ["writing_team"].
    """
    if interrupt_before is None:
        interrupt_before = ["writing_team"]

    research_graph = build_research_subgraph()
    writing_graph = build_writing_subgraph()

    super_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are the Top-Level Supervisor. You manage the research_team and writing_team to write a blog post. "
                   "First, route to the research_team to gather information. "
                   "Then, route to the writing_team to write the post based on the research. "
                   "When the final blog post is fully complete and presented, respond with FINISH."),
        MessagesPlaceholder(variable_name="messages"),
        ("system", "Given the conversation above, who should act next? Select one of: research_team, writing_team, FINISH")
    ])
    super_supervisor = super_prompt | llm.with_structured_output(SuperRoute)

    super_builder = StateGraph(WorkflowState)
    super_builder.add_node("supervisor", super_supervisor)
    super_builder.add_node("research_team", research_graph)
    super_builder.add_node("writing_team", writing_graph)

    super_builder.add_edge("research_team", "supervisor")
    super_builder.add_edge("writing_team", "supervisor")
    super_builder.add_conditional_edges(
        "supervisor",
        lambda x: x["next"],
        {"FINISH": END, "research_team": "research_team", "writing_team": "writing_team"}
    )
    super_builder.add_edge(START, "supervisor")

    return super_builder.compile(
        checkpointer=checkpointer,
        interrupt_before=interrupt_before
    )


# --- 5. High-Level HITL Workflow Manager ---
class HITLWorkflowManager:
    """Production manager for executing, inspecting, and resuming LangGraph agent graphs."""

    def __init__(self, db_path: str = "checkpoints.db", interrupt_before: Optional[List[str]] = None):
        self.db_path = db_path
        self.interrupt_before = interrupt_before or ["writing_team"]
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.checkpointer = SqliteSaver(self._conn)
        self.graph = build_hitl_workflow(
            checkpointer=self.checkpointer,
            interrupt_before=self.interrupt_before
        )

    def close(self):
        """Close SQLite database connection."""
        if self._conn:
            self._conn.close()

    def get_config(self, thread_id: str) -> Dict[str, Any]:
        """Generate LangGraph configuration dict with thread ID."""
        return {"configurable": {"thread_id": thread_id}}

    def start_workflow(
        self,
        thread_id: str,
        prompt: str,
        recursion_limit: int = 20
    ) -> Dict[str, Any]:
        """Start a new workflow run until it finishes or pauses at a breakpoint.
        
        Returns:
            Dict containing status ('PAUSED' or 'COMPLETED'), next node, and message count.
        """
        config = self.get_config(thread_id)
        config["recursion_limit"] = recursion_limit
        initial_input = {"messages": [HumanMessage(content=prompt)]}

        events = []
        for event in self.graph.stream(initial_input, config):
            events.append(event)

        state = self.graph.get_state(config)
        is_paused = bool(state.next)

        return {
            "thread_id": thread_id,
            "status": "PAUSED" if is_paused else "COMPLETED",
            "next": list(state.next) if state.next else [],
            "checkpoint_id": state.config.get("configurable", {}).get("checkpoint_id"),
            "messages_count": len(state.values.get("messages", []))
        }

    def inspect_state(self, thread_id: str) -> Dict[str, Any]:
        """Inspect current checkpoint state for a thread."""
        config = self.get_config(thread_id)
        state = self.graph.get_state(config)
        messages = state.values.get("messages", [])

        summary_messages = []
        for msg in messages:
            name = getattr(msg, "name", None) or type(msg).__name__
            summary_messages.append({
                "sender": name,
                "content_preview": msg.content[:160] if hasattr(msg, "content") else str(msg)[:160]
            })

        return {
            "thread_id": thread_id,
            "is_paused": bool(state.next),
            "next_node": list(state.next) if state.next else [],
            "checkpoint_id": state.config.get("configurable", {}).get("checkpoint_id"),
            "messages": summary_messages,
            "raw_state": state.values
        }

    def inject_feedback(
        self,
        thread_id: str,
        feedback: str,
        as_node: Optional[str] = None
    ) -> None:
        """Inject human steering feedback or constraints into the state at a breakpoint."""
        config = self.get_config(thread_id)
        feedback_msg = HumanMessage(
            content=f"[HUMAN_SUPERVISOR_FEEDBACK]: {feedback}",
            name="human_supervisor"
        )
        self.graph.update_state(
            config,
            {"messages": [feedback_msg], "human_feedback": feedback},
            as_node=as_node
        )

    def resume_workflow(
        self,
        thread_id: str,
        feedback: Optional[str] = None,
        recursion_limit: int = 20
    ) -> Dict[str, Any]:
        """Resume execution of a paused workflow from the saved breakpoint checkpoint."""
        if feedback:
            self.inject_feedback(thread_id, feedback)

        config = self.get_config(thread_id)
        config["recursion_limit"] = recursion_limit

        events = []
        for event in self.graph.stream(None, config):
            events.append(event)

        state = self.graph.get_state(config)
        is_paused = bool(state.next)

        return {
            "thread_id": thread_id,
            "status": "PAUSED" if is_paused else "COMPLETED",
            "next": list(state.next) if state.next else [],
            "checkpoint_id": state.config.get("configurable", {}).get("checkpoint_id"),
            "messages_count": len(state.values.get("messages", []))
        }

    def get_history(self, thread_id: str) -> List[Dict[str, Any]]:
        """Retrieve audit history of all checkpoints saved for a thread."""
        config = self.get_config(thread_id)
        history = []
        for state in self.graph.get_state_history(config):
            cfg = state.config.get("configurable", {})
            history.append({
                "checkpoint_id": cfg.get("checkpoint_id"),
                "checkpoint_ns": cfg.get("checkpoint_ns"),
                "next": list(state.next) if state.next else [],
                "messages_count": len(state.values.get("messages", []))
            })
        return history


# --- 6. Interactive CLI Entrypoint ---
def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Human-in-the-Loop Breakpoints & SQLite State Checkpointing"
    )
    parser.add_argument("--thread-id", type=str, default="demo-thread-1", help="Thread ID for state persistence")
    parser.add_argument("--db-path", type=str, default="checkpoints.db", help="SQLite database path")
    parser.add_argument("--prompt", type=str, help="Initial prompt to trigger workflow")
    parser.add_argument("--inspect", action="store_true", help="Inspect current checkpoint state")
    parser.add_argument("--feedback", type=str, help="Inject human guidance or constraints")
    parser.add_argument("--resume", action="store_true", help="Resume execution from paused breakpoint")
    parser.add_argument("--history", action="store_true", help="Display thread checkpoint history audit log")

    args = parser.parse_args()

    manager = HITLWorkflowManager(db_path=args.db_path)

    try:
        if args.inspect:
            info = manager.inspect_state(args.thread_id)
            print(f"\n🔍 State Inspection for Thread '{args.thread_id}':")
            print(f"  - Paused at Breakpoint: {info['is_paused']}")
            print(f"  - Next Node: {info['next_node']}")
            print(f"  - Checkpoint ID: {info['checkpoint_id']}")
            print(f"  - Total Messages: {len(info['messages'])}")
            print("\nRecent Conversation Log:")
            for m in info["messages"][-5:]:
                print(f"  [{m['sender']}]: {m['content_preview']}...")
            return

        if args.history:
            history = manager.get_history(args.thread_id)
            print(f"\n📜 Checkpoint History Audit Log for Thread '{args.thread_id}' ({len(history)} checkpoints):")
            for idx, h in enumerate(history):
                print(f"  #{idx+1} [ID: {h['checkpoint_id'][:8]}...] Next: {h['next']} | Msgs: {h['messages_count']}")
            return

        if args.resume:
            print(f"\n▶️ Resuming workflow for Thread '{args.thread_id}'...")
            res = manager.resume_workflow(args.thread_id, feedback=args.feedback)
            print(f"Result: Status={res['status']}, Next={res['next']}, Total Msgs={res['messages_count']}")
            return

        # Start workflow
        prompt = args.prompt or "Analyze how Human-in-the-Loop breakpoints prevent hallucinated content in AI writing pipelines."
        print(f"\n🚀 Starting HITL Checkpointed Workflow (Thread: {args.thread_id})...")
        print(f"Prompt: {prompt}\n")
        res = manager.start_workflow(args.thread_id, prompt)
        print(f"\n⏸️  Workflow Status: {res['status']}")
        if res["status"] == "PAUSED":
            print(f"   Paused before node(s): {res['next']}")
            print(f"   Checkpoint persisted to '{args.db_path}' (ID: {res['checkpoint_id']})")
            print("\n💡 You can now:")
            print(f"   1. Inspect state:  python hitl_checkpoint_workflow.py --thread-id {args.thread_id} --inspect")
            print(f"   2. Resume & approve: python hitl_checkpoint_workflow.py --thread-id {args.thread_id} --resume")
            print(f"   3. Steer with feedback: python hitl_checkpoint_workflow.py --thread-id {args.thread_id} --resume --feedback 'Focus on banking compliance'")
        else:
            print(f"   Workflow completed successfully!")
    finally:
        manager.close()


if __name__ == "__main__":
    main()
