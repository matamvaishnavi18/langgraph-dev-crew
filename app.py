%%writefile app.py
import sys
import io
import traceback
from typing import TypedDict, List, Optional

from fastapi import FastAPI
from pydantic import BaseModel

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langserve import add_routes

import os
import google.generativeai as genai

# ====================================================
# GOOGLE API KEY & MODEL INITIALIZATION
# ====================================================

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable not found.")

genai.configure(api_key=GOOGLE_API_KEY)

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.2
)

# ====================================================
# STATE
# ====================================================

class CrewState(TypedDict):
    messages: List[BaseMessage]
    next_step: Optional[str]
    code: Optional[str]
    report: Optional[str]

# ====================================================
# TOOLS
# ====================================================

@tool
def run_python_code(code: str) -> str:
    """Execute Python code."""
    clean_code = (
        code.replace("```python", "")
        .replace("```", "")
        .strip()
    )

    old_stdout = sys.stdout
    new_stdout = io.StringIO()
    sys.stdout = new_stdout

    try:
        exec(clean_code, {})
        result = new_stdout.getvalue()
    except Exception:
        result = traceback.format_exc()
    finally:
        sys.stdout = old_stdout

    return result if result else "Success (No Output)"


@tool
def generate_test_cases(task_description: str) -> str:
    """Generate test cases."""
    prompt = f"""
Generate 3-5 Python test scenarios for:

{task_description}

Return only numbered list.
"""
    response = llm.invoke(prompt)
    return response.content if hasattr(response, "content") else str(response)

# ====================================================
# NODES
# ====================================================

def developer_node(state: CrewState):
    task = state["messages"][-1].content
    prompt = f"""
Write clean Python code for:

{task}

Return only code.
"""
    response = llm.invoke(prompt)
    code = response.content if isinstance(response.content, str) else str(response.content)
    return {"code": code}


def tester_node(state: CrewState):
    task = state["messages"][-1].content
    tests = generate_test_cases.invoke({"task_description": task})
    output = run_python_code.invoke({"code": state["code"]})

    report = f"""
### Generated Code

{state["code"]}

--------------------

### Execution

{output}

--------------------

### Test Cases

{tests}
"""
    return {"report": report}

# ====================================================
# GRAPH
# ====================================================

graph = StateGraph(CrewState)

graph.add_node("developer", developer_node)
graph.add_node("tester", tester_node)

graph.add_edge(START, "developer")
graph.add_edge("developer", "tester")
graph.add_edge("tester", END)

workflow = graph.compile()

# ====================================================
# FASTAPI & LANGSERVE ROUTES
# ====================================================

app = FastAPI(title="AI Coding Crew")

@app.get("/")
def home():
    return {"message": "AI Coding Crew Running"}

# Registers /agent and /agent/playground/
add_routes(app, workflow, path="/agent")
