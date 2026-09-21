import os
import sys
from typing import Annotated
from typing_extensions import TypedDict
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

# Attempt to load environment variables from a local .env file if available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# 1. Define the State
# The state is a dictionary that will be passed between nodes.
# `add_messages` appends new messages to the existing list instead of overwriting it.
class State(TypedDict):
    messages: Annotated[list, add_messages]

# 2. Define the LLM and Nodes
# Set your GEMINI_API_KEY environment variable or populate .env
api_key = os.environ.get("GEMINI_API_KEY")
model_name = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

llm = ChatGoogleGenerativeAI(
    model=model_name,
    google_api_key=api_key or "placeholder_key"
)

def chatbot(state: State):
    """The node function that invokes the LLM with the current message history."""
    response = llm.invoke(state["messages"])
    # Return the new message to be appended to the state
    return {"messages": [response]}

# 3. Build the Graph
graph_builder = StateGraph(State)

# Add our node to the graph
graph_builder.add_node("chatbot", chatbot)

# Define the flow: START -> chatbot -> END
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)

# Compile the graph into a runnable LangChain runnable
def compile_graph(checkpointer=None):
    """Compile graph with optional checkpointer (e.g. SqliteSaver)."""
    return graph_builder.compile(checkpointer=checkpointer)

graph = compile_graph()

# 4. Run the Agent
if __name__ == "__main__":
    print(f"Simple LangGraph Gemini Agent Started! (Model: {model_name})")
    print("Type 'quit', 'exit', or 'q' to stop.\n")
    
    if not api_key:
        print("⚠️  WARNING: GEMINI_API_KEY is not set.")
        print("   Set it in your terminal: export GEMINI_API_KEY='your-key-here'")
        print("   Or create a .env file (see .env.example)\n")

    while True:
        try:
            user_input = input("User: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break
            
        # Stream the graph execution
        try:
            for event in graph.stream({"messages": [user_input]}):
                for value in event.values():
                    print("Assistant:", value["messages"][-1].content)
        except Exception as e:
            print(f"Error executing agent turn: {e}")
