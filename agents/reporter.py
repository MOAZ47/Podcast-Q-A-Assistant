import sys, os, time, logging
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_cohere import ChatCohere

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import config

# --- Logging ---
LOG_DIR = os.path.join(os.getcwd(), "logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_PATH = os.path.join(LOG_DIR, "report.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH),
        logging.StreamHandler()
    ]
)

# Setup Logging
logger = logging.getLogger(__name__)

def generate_final_report(summary: str, fact_check_output: str) -> str:
    """
    Generate a markdown-formatted final report from a podcast summary and its fact check result.
    """

    # Model
    llm = ChatCohere(model="command-r", cohere_api_key=config.COHERE_API_KEY)

    # --- Examples ---
    example1 = {
        "summary": "The host stated that WHO declared COVID-19 a pandemic in March 2020 and that vaccines became widely available in early 2021.",
        "fact_check": (
            "Confirmed: WHO declared COVID-19 a global pandemic on March 11, 2020.\n"
            "Confirmed: Vaccines became widely available to the public in early 2021 in the US and EU.\n"
            "No major inaccuracies found."
        ),
        "final_report": """### ✅ Fact Check Report

        **Confirmed Claims:**
        - WHO declared COVID-19 a pandemic in March 2020.
        - Vaccines were widely available by early 2021.

        **Inaccuracies:** None found.

        **Confidence Level:** High  
        **Recommendation:** The summary is factually reliable and can be trusted.
        """
    }

    example2 = {
        "summary": "The speaker claimed that electric vehicles (EVs) never require battery replacement.",
        "fact_check": (
            "Inaccuracy: EV batteries degrade and may need replacement after 8–10 years.\n"
            "Partially accurate: Some EVs are long-lasting, but 'never' is misleading."
        ),
        "final_report": """### ⚠️ Fact Check Report

        **Confirmed Claims:**
        - EV batteries can last many years without replacement.

        **Inaccuracies:**
        - Claiming EVs 'never' need battery replacement is misleading. Battery degradation occurs over time.

        **Confidence Level:** Medium  
        **Recommendation:** Be cautious. Some misleading information is present.
        """
    }

    # --- Prompt Templates ---
    example_prompt = PromptTemplate(
        input_variables=["summary", "fact_check", "final_report"],
        template=(
            "Summary:\n{summary}\n\n"
            "Fact Check Results:\n{fact_check}\n\n"
            "Final Report:\n{final_report}"
        )
    )

    fewshot_prompt = FewShotPromptTemplate(
        examples=[example1, example2],
        example_prompt=example_prompt,
        prefix=(
            "You are a fact-checking report generator. Based on the summary and fact-check output, generate a markdown-formatted report using this structure:\n\n"
            "### [Emoji] Fact Check Report\n\n"
            "**Confirmed Claims:**\n- ...\n\n"
            "**Inaccuracies:**\n- ...\n\n"
            "**Confidence Level:** ...\n"
            "**Recommendation:** ..."
        ),
        suffix=(
            "\n---\nNow evaluate this:\n\n"
            "Summary:\n"
            "{summary}\n\n"
            "Fact Check Results:\n"
            "{fact_check}\n\n"
            "Please follow these instructions:\n"
            "- Use proper markdown formatting.\n"
            "- Clearly list confirmed claims.\n"
            "- Call out any inaccuracies.\n"
            "- Include a confidence level (High, Medium, Low).\n"
            "- State whether the summary can be trusted.\n\n"
            "Final Report:"
        ),
        input_variables=["summary", "fact_check"]
    )

    # Run the chain
    chain = (
        {"summary": lambda _: summary.strip(), "fact_check": lambda _: fact_check_output.strip()}
        | fewshot_prompt
        | llm
        | StrOutputParser()
    )

    final_report = chain.invoke({"summary": summary, "fact_check": fact_check_output})

    # --- Optional Warning ---
    if not final_report.strip().startswith("###"):
        logger.warning("[WARNING] Final report is not in expected markdown format.")

    return final_report



# --- CLI test ---
if __name__ == "__main__":
    sample_summary = (
        "Tesla CEO Elon Musk claimed during the podcast that their cars can fully drive themselves without human intervention as of 2023."
    )
    sample_fact_check = (
        "Inaccuracy: As of 2023, Tesla's Full Self-Driving (FSD) software requires driver supervision and is not fully autonomous.\n"
        "Partially accurate: Tesla has made progress, but Level 5 autonomy is not yet achieved.\n"
        "No regulatory approval exists for full self-driving as of 2023."
    )

    print("📄 Generated Report:\n")
    print(generate_final_report(sample_summary, sample_fact_check))
