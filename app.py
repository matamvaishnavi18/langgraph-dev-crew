import sys
import io
import os
import asyncio
import traceback
from typing import TypedDict, List, Optional

from fastapi import FastAPI
from pydantic import BaseModel, Field

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from langserve import add_routes
import google.generativeai as genai


# ============================================================
# 1. API KEY & MODEL INITIALIZATION
# ============================================================

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable missing.")

genai.configure(api_key=GOOGLE_API_KEY)

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.2
)


# ============================================================
# 2. SCHEMAS (INPUT & OUTPUT)
# ============================================================

class TaskInput(BaseModel):
    task: str = Field(..., description="Programming task description")

class TaskOutput(BaseModel):
    report: str = Field(..., description="Final execution & test report")

class CrewState(TypedDict):
    messages: List[BaseMessage]
    code: Optional[str]
    report: Optional[str]


# ============================================================
# 3. HELPER FUNCTIONS & NODES
# ============================================================

def run_python_code(code: str) -> str:
    clean_code = code.replace("```python", "").replace("```", "").strip()
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

    return result if result.strip() else "Success (No Output / Printed Statements)"


async def generate_test_cases(task_description: str) -> str:
    prompt = f"Generate 3-5 Python test scenarios for:\n{task_description}\nReturn only numbered list."
    response = await llm.ainvoke(prompt)
    return response.content if hasattr(response, "content") else str(response)


async def developer_node(state: CrewState):
    messages = state.get("messages", [])
    task = messages[-1].content if messages else "No task provided."
    prompt = f"Write clean, executable Python code for:\n{task}\nReturn ONLY Python code inside backticks."
    response = await llm.ainvoke(prompt)
    code = response.content if isinstance(response.content, str) else str(response.content)
    return {"code": code}


async def tester_node(state: CrewState):
    messages = state.get("messages", [])
    task = messages[-1].content if messages else "No task provided."
    tests = await generate_test_cases(task)
    output = run_python_code(state.get("code", ""))

    report = (
        f"### Generated Code\n{state.get('code', '')}\n\n"
        f"---\n\n### Execution Output\n{output}\n\n"
        f"---\n\n### Test Cases\n{tests}"
    )
    return {"report": report}


# ============================================================
# 4. BUILD LANGGRAPH & RUNNABLE CHAIN
# ============================================================

graph = StateGraph(CrewState)
graph.add_node("developer", developer_node)
graph.add_node("tester", tester_node)
graph.add_edge(START, "developer")
graph.add_edge("developer", "tester")
graph.add_edge("tester", END)
workflow = graph.compile()

# Extract only the "report" key so LangServe can render it in the main Output box
runnable_chain = (
    (lambda x: {"messages": [HumanMessage(content=x["task"])]}) 
    | workflow 
    | (lambda state: {"report": state.get("report", "No report generated.")})
)


# ============================================================
# 5. FASTAPI & LANGSERVE ROUTES
# ============================================================

app = FastAPI(title="AI Coding Crew")

@app.get("/")
def home():
    return {"message": "AI Coding Crew Running"}

add_routes(
    app,
    runnable_chain.with_types(input_type=TaskInput, output_type=TaskOutput),
    path="/agent"
)


# ============================================================
# 6. SERVER ENTRY POINT
# ============================================================

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
