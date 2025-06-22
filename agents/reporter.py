from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_cohere import ChatCohere
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import config

def reporter(summary, fact_check_output):
    llm = ChatCohere(model="command", cohere_api_key=config.COHERE_API_KEY)
    # Example 1
    example1 = {
        "summary": "Elon Musk discussed reusable rockets and their role in making Mars colonization affordable.",
        "fact_check": "Confirmed: SpaceX is developing reusable Starship rockets. Confirmed: Mars colonization is a stated goal. No inaccuracies found.",
        "final_report": """Confirmed claims include the development of reusable rockets by SpaceX and the goal of colonizing Mars. No factual inaccuracies were found.
    Confidence Level: High
    Recommendation: The summary can be trusted."""
    }

    # Example 2
    example2 = {
        "summary": "The speaker claimed that AI is already sentient and can feel emotions.",
        "fact_check": "Partially confirmed: AI systems can simulate emotions but are not sentient. Inaccuracy: No current AI is proven to be sentient.",
        "final_report": """The claim that AI is sentient is inaccurate, though it's true that AI can simulate emotions. This weakens the reliability of the summary.
    Confidence Level: Medium
    Recommendation: Use caution when trusting this summary."""
    }

    
    example_prompt = PromptTemplate(
        input_variables=["summary", "fact_check", "final_report"],
        template="""
            Summary:
            {summary}

            Fact Check Results:
            {fact_check}

            Final Report:
            {final_report}
        """
    )
    
    report_prompt = FewShotPromptTemplate(
        examples=[example1, example2],
        example_prompt=example_prompt,
        prefix=(
            "You are a final reporting agent. Your job is to review a podcast summary and fact-checking output, "
            "then write a final report with your assessment.\n\n"
            "Below are a few example reports:\n"
        ),
        suffix=(
            "---\n"
            "Now analyze the following:\n\n"
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

    report_chain = (
        {"summary": lambda x: summary, "fact_check": lambda x: fact_check_output}
        | report_prompt
        | llm
        | StrOutputParser()
    )

    response = report_chain.invoke({"summary": summary, "fact_check": fact_check_output})
    # print(f"Report: \n{response} \n")
    return response
    
if __name__ == "__main__":
    summary = "Space X has undertaken another test launch of a giant new rocket that it calls Starship. Starship lifted off just after 7.30pm Eastern time today. But not everything has been going according to plan. Empire Science correspondent Jeff Brumfield joins us now to talk about this latest attempt. SpaceX said it had sprung a fuel leak and I was watching it tumble back in above the Indian Ocean. SpaceX later said that the problem there was sort of some harmonic response in the first launch. That's just a wicked vibration that actually shook up the engines until they broke and then in March there was a hardware failure and a single engine. SpaceX made it to space but you know you can't call this a success. This program is starting to look like it's slipping behind. Starship was supposed to be able to at least orbit the earth by now. And on this particular flight the fact they couldn't hit reentry is a big problem. SpaceX has talked about sending a Starship without people to Mars as soon as next year. There's so much they need to work through to get the spacecraft working. NASA wants Starship to land people on the moon as soon as 2027. But you never count Elon Musk or SpaceX out."

    fact_check_output = "The article alleges the following points:\n1. Starship is a rocket developed by SpaceX that underwent a test launch mentioned in the statement.\n2. The launch time stated was incorrect, as it lifted off just after 7.30 pm Eastern time.\n3. Problems during the launch included a fuel leak and vibrations that shook the engines causing a malfunction. These issues caused the rocket to fail to orbit Earth as planned.\n4. SpaceX has previously announced plans to send a Starship spacecraft to Mars without passengers as soon as next year.\n5. NASA wants to use Starship to land personnel on the moon by 2027.\nAfter researching, I can confirm the factual accuracy of the statement with the exception of item number 2 which should read as follows: \"Starship lifted off from SpaceX's Starbase facility in Boca Chica, Texas, at approximately 2:54 pm CST on November 6, 2022.\"\nLet me know if you'd like me to verify any other statements."

    report = reporter(summary, fact_check_output)
    print("\n📄 Final Report:\n")
    print(report)