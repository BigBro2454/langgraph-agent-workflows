import os
import sys
from typing import Annotated, Literal, TypedDict
from pydantic import BaseModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

# Attempt to load environment variables from a local .env file if available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Set up LLM
api_key = os.environ.get("GEMINI_API_KEY")
model_name = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

llm = ChatGoogleGenerativeAI(
    model=model_name,
    google_api_key=api_key or "placeholder_key"
)

# Global State
class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    next: str

# --- 1. Research Team Subgraph ---
class ResearchRoute(BaseModel):
    next: Literal["search_agent", "web_scraper_agent", "FINISH"]

research_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are the supervisor of the Research Team. Your goal is to gather information. "
               "Your team members are: search_agent, web_scraper_agent. "
               "When you have gathered enough information, respond with FINISH."),
    MessagesPlaceholder(variable_name="messages"),
    ("system", "Given the conversation above, who should act next? Select one of: search_agent, web_scraper_agent, FINISH")
])
research_supervisor = research_prompt | llm.with_structured_output(ResearchRoute)

def search_agent_node(state: State):
    response = llm.invoke([SystemMessage(content="You are a search agent. Find relevant information on the topic.")] + state["messages"])
    return {"messages": [HumanMessage(content=response.content, name="search_agent")]}

def web_scraper_agent_node(state: State):
    response = llm.invoke([SystemMessage(content="You are a web scraper agent. Extract detailed data to support the research.")] + state["messages"])
    return {"messages": [HumanMessage(content=response.content, name="web_scraper_agent")]}

research_builder = StateGraph(State)
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
research_graph = research_builder.compile()

# --- 2. Writing Team Subgraph ---
class WritingRoute(BaseModel):
    next: Literal["doc_writer_agent", "note_taker_agent", "chart_generator_agent", "FINISH"]

writing_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are the supervisor of the Writing Team. Your goal is to compile a report based on the research. "
               "Your team members are: doc_writer_agent, note_taker_agent, chart_generator_agent. "
               "When the blog post is complete, respond with FINISH."),
    MessagesPlaceholder(variable_name="messages"),
    ("system", "Given the conversation above, who should act next? Select one of: doc_writer_agent, note_taker_agent, chart_generator_agent, FINISH")
])
writing_supervisor = writing_prompt | llm.with_structured_output(WritingRoute)

def doc_writer_agent_node(state: State):
    response = llm.invoke([SystemMessage(content="You are a doc writer agent. Write the final blog post incorporating research notes.")] + state["messages"])
    return {"messages": [HumanMessage(content=response.content, name="doc_writer_agent")]}

def note_taker_agent_node(state: State):
    response = llm.invoke([SystemMessage(content="You are a note taker agent. Organize the research findings into coherent bullet points.")] + state["messages"])
    return {"messages": [HumanMessage(content=response.content, name="note_taker_agent")]}

def chart_generator_agent_node(state: State):
    response = llm.invoke([SystemMessage(content="You are a chart generating agent. Create markdown tables representing key data points.")] + state["messages"])
    return {"messages": [HumanMessage(content=response.content, name="chart_generator_agent")]}

writing_builder = StateGraph(State)
writing_builder.add_node("supervisor", writing_supervisor)
writing_builder.add_node("doc_writer_agent", doc_writer_agent_node)
writing_builder.add_node("note_taker_agent", note_taker_agent_node)
writing_builder.add_node("chart_generator_agent", chart_generator_agent_node)

for member in ["doc_writer_agent", "note_taker_agent", "chart_generator_agent"]:
    writing_builder.add_edge(member, "supervisor")

writing_builder.add_conditional_edges(
    "supervisor",
    lambda x: x["next"],
    {"FINISH": END, "doc_writer_agent": "doc_writer_agent", "note_taker_agent": "note_taker_agent", "chart_generator_agent": "chart_generator_agent"}
)
writing_builder.add_edge(START, "supervisor")
writing_graph = writing_builder.compile()

# --- 3. Top-Level Supervisor ---
class SuperRoute(BaseModel):
    next: Literal["research_team", "writing_team", "FINISH"]

super_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are the Top-Level Supervisor. You manage the research_team and writing_team to write a blog post. "
               "First, route to the research_team to gather information. "
               "Then, route to the writing_team to write the post based on the research. "
               "When the final blog post is fully complete and presented, respond with FINISH."),
    MessagesPlaceholder(variable_name="messages"),
    ("system", "Given the conversation above, who should act next? Select one of: research_team, writing_team, FINISH")
])
super_supervisor = super_prompt | llm.with_structured_output(SuperRoute)

super_builder = StateGraph(State)
super_builder.add_node("supervisor", super_supervisor)
# LangGraph allows adding compiled graphs directly as nodes
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
super_graph = super_builder.compile()

# --- 4. Run Execution ---
if __name__ == "__main__":
    if not api_key:
        print("⚠️  WARNING: GEMINI_API_KEY is not set.")
        print("   Set it via: export GEMINI_API_KEY='your-key-here'")
        print("   Or create a .env file (see .env.example)\n")
        
    print(f"🚀 Hierarchical Multi-Agent Supervisor System Started! (Model: {model_name})")
    
    user_input = sys.argv[1] if len(sys.argv) > 1 else "Write a comprehensive technical overview about the benefits of multi-agent systems in enterprise AI architectures."
    print(f"User Request: {user_input}\n")
    print("-" * 60)
    
    try:
        events = super_graph.stream(
            {"messages": [HumanMessage(content=user_input)]},
            {"recursion_limit": 20}
        )
        
        for event in events:
            for node_name, node_state in event.items():
                if isinstance(node_state, dict):
                    if "messages" in node_state and node_state["messages"]:
                        latest_message = node_state["messages"][-1]
                        name = getattr(latest_message, "name", None) or node_name
                        preview = latest_message.content[:240].strip()
                        print(f"[{node_name}] {name}:\n{preview}...\n")
                    elif "next" in node_state:
                        print(f"[{node_name}] Routing to -> {node_state['next']}\n")
        print("-" * 60)
        print("✅ Workflow execution finished successfully.")
    except Exception as e:
        print(f"Execution terminated with: {e}")
