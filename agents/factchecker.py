from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import Tool
from langchain_cohere import ChatCohere

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import config


def fact_check(summary):
    # Step 3: Agent 2 – Fact Checker (Using Web Search)
    llm = ChatCohere(model= 'command', cohere_api_key=config.COHERE_API_KEY)

    search = DuckDuckGoSearchRun()

    tools = [
        Tool(
            name="DuckDuckGo Search",
            func=search.run,
            description="Search the web for facts mentioned in a podcast transcript."
        )
    ]

    fact_check_prompt = ChatPromptTemplate.from_template(
        """Check the following summary for factual accuracy:

        Summary: {summary}

        Search the web and list any factual inconsistencies or confirmations. Be concise.
        """
    )

    fact_check_chain = (
        {"summary": RunnablePassthrough()}
        | fact_check_prompt
        | llm
        | StrOutputParser()
    )

    return fact_check_chain.invoke(summary)

if __name__ == "__main__":
    summary = "Space X has undertaken another test launch of a giant new rocket that it calls Starship. Starship lifted off just after 7.30pm Eastern time today. But not everything has been going according to plan. Empire Science correspondent Jeff Brumfield joins us now to talk about this latest attempt. SpaceX said it had sprung a fuel leak and I was watching it tumble back in above the Indian Ocean. SpaceX later said that the problem there was sort of some harmonic response in the first launch. That's just a wicked vibration that actually shook up the engines until they broke and then in March there was a hardware failure and a single engine. SpaceX made it to space but you know you can't call this a success. This program is starting to look like it's slipping behind. Starship was supposed to be able to at least orbit the earth by now. And on this particular flight the fact they couldn't hit reentry is a big problem. SpaceX has talked about sending a Starship without people to Mars as soon as next year. There's so much they need to work through to get the spacecraft working. NASA wants Starship to land people on the moon asSoon as 2027. But you never count Elon Musk or SpaceX out."
    fac = fact_check(summary)
    print(f"Type of fact check = {type(fac)}")
    print(f"Response of fact check = {fac}")