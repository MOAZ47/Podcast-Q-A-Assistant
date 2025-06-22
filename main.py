from agents.transcription import transcribe_podcast
from agents.summarizer import summarize_text
from agents.factchecker import fact_check
from agents.reporter import reporter

from langchain.agents import Tool
from langchain_cohere import ChatCohere

import config



def initialize_pipeline (podcast_transcript):

    summarizer_tool = Tool(
        name="PodcastSummarizer",
        func=summarize_text,
        description="Summarize the podcast transcription."
    )

    
    if podcast_transcript:
        summary = summarizer_tool.run(podcast_transcript)

    fact_check_output = fact_check(summary)

    print(f"TYPE Fact Check Output: \n{type(fact_check_output)} \n")

    report = reporter(summary, fact_check_output)

    #print(f"Report: \n{report} \n")

    return report