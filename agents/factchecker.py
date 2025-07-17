import os
import sys
import logging
import warnings
import time
from datetime import datetime
import pytz
from concurrent.futures import ThreadPoolExecutor

from langchain.agents import AgentExecutor, create_tool_calling_agent, Tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_cohere import ChatCohere
from tavily import TavilyClient

# --- Path setup ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config

# --- Logging Setup ---
LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_PATH = os.path.join(LOG_DIR, "factchecker.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- Suppress known warnings ---
warnings.filterwarnings("ignore", category=DeprecationWarning, module="pydantic")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="cohere")
warnings.filterwarnings("ignore", category=FutureWarning, module="cohere.core.unchecked_base_model")

# --- Initialize LLM ---
llm = ChatCohere(
    model="command-r",
    temperature=0.3,
    cohere_api_key=config.COHERE_API_KEY
)

# --- Tavily API Setup ---
TAVILY_API_KEY = getattr(config, "TAVILY_API_KEY", os.getenv("TAVILY_API_KEY"))
tavily = TavilyClient(api_key=TAVILY_API_KEY)

# --- Parallel Search Functions (Threaded) ---
def threaded_search(query: str) -> str:
    logger.info(f"[SEARCH] Querying: {query}")
    try:
        result = tavily.search(query=query, include_answer=True, max_results=3)
        answer = result.get("answer") or "No clear answer found."
        logger.info(f"[RESULT] {query} => {answer[:100]}...")
        return answer
    except Exception as e:
        logger.error(f"[ERROR] Search failed for '{query}': {e}")
        return "Search error."

def run_parallel_searches(queries: list[str]) -> dict[str, str]:
    logger.info(f"[INFO] Running {len(queries)} threaded searches.")
    with ThreadPoolExecutor(max_workers=5) as executor:
        results = list(executor.map(threaded_search, queries))
    return dict(zip(queries, results))

# --- LangChain Tool Wrapper ---
class ParallelFactSearchTool:
    name = "parallel_fact_search"
    description = "Run multiple fact-checking searches in parallel and return structured answers."

    def run(self, queries: str) -> str:
        query_list = [q.strip() for q in queries.strip().split("\n") if q.strip()]
        if not query_list:
            logger.warning("[WARNING] No valid search queries received.")
            return "No valid queries provided."

        results = run_parallel_searches(query_list)
        return "\n".join(f"{q}: {a}" for q, a in results.items())

# --- Tool Registration ---
tools = [
    Tool(
        name="parallel_fact_search",
        func=ParallelFactSearchTool().run,
        description="Useful for checking multiple factual claims at once. Input should be newline-separated search questions."
    )
]

# --- Prompt Template ---
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a highly accurate fact-checking assistant. Your job is to verify claims.

Follow these steps:
1. Break down complex input into specific factual sub-claims.
2. Formulate precise search questions.
3. Use `parallel_fact_search` to batch queries (newline-separated).
4. Analyze answers from reliable sources.
5. Provide a verdict: "Factually Accurate", "Partially Accurate", or "Inaccurate".
6. Support your verdict with clear evidence and sources (e.g. [Source: NASA.gov]).

Current Date and Time: {current_datetime}
Current Location: {current_location}
"""),
    ("human", "{input}"),
    MessagesPlaceholder("agent_scratchpad")
])

# --- Agent + Executor Setup ---
agent = create_tool_calling_agent(llm=llm, tools=tools, prompt=prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

# --- Main API ---
def fact_check(claim: str) -> str:
    try:
        start_check = time.time()
        logger.info(f"[FACT CHECK] Started for input: {claim[:100]}...")
        ist = pytz.timezone('Asia/Kolkata')
        now = datetime.now(ist).strftime("%A, %B %d, %Y at %I:%M:%S %p %Z")
        location = "Mumbai, Maharashtra, India"

        result = agent_executor.invoke({
            "input": claim,
            "current_datetime": now,
            "current_location": location
        })

        output = result.get("output", "")
        end_check = time.time()
        logger.info("[FACT CHECK] Completed successfully.")
        logger.info(f"[DONE] Fact Check completed in {end_check - start_check:.2f}s")
        return output
    except Exception as e:
        logger.error(f"[ERROR] Fact-checking failed: {e}")
        return f"Error: {str(e)}"

# --- Direct Test Run ---
if __name__ == "__main__":
    logger.info("[TEST] Running fact checker on test claims...")
    test_claim = """NASA is funding the Artemis program for a moon landing in 2025.
SpaceX launched Starship in 2023.
ISRO's Chandrayaan 3 landed successfully on the moon in 2023."""

    output = fact_check(test_claim)
    print("\n🔍 Final Output:\n", output)
