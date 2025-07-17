#from agents.transcription import transcribe_podcast
from agents.cohere_summarizer import summarize_text
from agents.factchecker import fact_check
from agents.reporter import generate_final_report

from langchain.agents import Tool
#from langchain_cohere import ChatCohere
import time, os, sys
import logging
from logging.handlers import RotatingFileHandler


# --- Logging ---
LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_PATH = os.path.join(LOG_DIR, "main.log")

# Get a logger specific to this module
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO) # Set the minimum logging level for this logger

# Check if handlers are already attached to prevent duplicate logs
# This is crucial in environments where the script might be reloaded (e.g., Streamlit)
if not logger.handlers:
    # Create a file handler for logging to a file
    file_handler = RotatingFileHandler(LOG_PATH, maxBytes=5_000_000, backupCount=2, encoding="utf-8")
    file_handler.setLevel(logging.INFO) # Set level for file output

    # Create a stream handler for logging to console (sys.stdout)
    stream_handler = logging.StreamHandler(stream=sys.stdout)
    stream_handler.setLevel(logging.INFO) # Set level for console output
    # Ensure console output also uses UTF-8, even without emojis, for broader compatibility
    stream_handler.stream.reconfigure(encoding='utf-8')

    # Define the log format
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # Apply the formatter to both handlers
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)

    # Add the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

def initialize_pipeline (podcast_transcript):
    start = time.time()
    logger.info("[START] Running analysis pipeline...")

    if not podcast_transcript or len(podcast_transcript.strip()) < 10:
        logger.error("[ERROR] Transcript too short or missing.")
        return "Error: Transcript is empty or invalid."

    # --- 1. Summarization ---
    t1 = time.time()
    summary = summarize_text(podcast_transcript)
    t2 = time.time()
    logger.info(f"[INFO] Summarization took {t2 - t1:.2f} seconds")


    # --- Fact Checking ---
    t3 = time.time()
    fact_check_output = fact_check(summary)
    t4 = time.time()
    logger.info(f"[INFO] Fact Check took {t4 - t3:.2f} seconds")


    # --- Report Generation ---
    t5 = time.time()
    report = generate_final_report(summary, fact_check_output)
    t6 = time.time()
    logger.info(f"[INFO] Report generation took {t6 - t5:.2f} seconds")

    total = time.time() - start
    logger.info(f"[INFO] Total pipeline time: {total:.2f} seconds")
    logger.info("[INFO] Report preview:\n" + report[:400] + "...\n")
    logger.info("[END] Report generated.")

    return report

if __name__ == "__main__":
    # 🔁 Test input: simulate transcript from app.py
    sample_transcript = (
    "All right, the commercial company Space X has undertaken another test launch of a giant new rocket that it calls Starship. \n"
    "Starship lifted off just after 7.30 p.m. Eastern time today. But not everything has been going according to plan. \n"
    "Joining us now to talk about this latest attempt and what is at stake is Empire Science correspondent Jeff Bromfield, hey Jeff. \n"
    "Not according to plan. Okay, so tell us, how is it going so far? Well, Starship took off from Starbase in Texas, Southern Texas. \n"
    "It flew out over the Gulf and separated from this enormous super heavy booster. Now, on previous launches, that booster has gone back to the launch pad and actually been caught by giant mechanical arms. \n"
    "But this time they wanted to test some emergency contingency features so they sent it out over the Gulf and it's a good thing they did because the engines did not relied on the booster as expected and the booster appeared to crash into the water. \n"
    "Now, Starship made it into space but there are some signs that things may not have gone according to plan there either. \n"
    "The ship started out in a very slow tumble which then sped up. SpaceX said it had sprung a fuel leak and I was watching it tumble back in above the Indian ocean. Very, very pretty rainbow colors as it burned up in the atmosphere but unfortunately not what SpaceX wanted to see. \n"
    "Right, it sounds pretty but kind of janky so would this test be considered a success or a failure or something in between what do you think? I'm not sure myself honestly. I mean, the booster didn't matter. They were testing that to the point of failure anyway but here's the problem. \n"
    "They had some launches in January and March with both of which ended up with Starship exploding before it could reach space. Now, SpaceX later said that the problem there was sort of some harmonic response in the first launch. That's just a wicked vibration that actually shook up the engines until they broke. \n"
    "And then in March, there was a hardware failure and a single engine that started to fire on Starship. So by those standards, today's flight was better because they did make it to space but I don't think you can call this a success. They made it up but not back down. Right. Okay, well how much was writing on this one launch today? \n"
    "You know, SpaceX will say, look, this is just a test launch. They do expect failures to happen. And this is how the company works. They iterate, they redesign. But honestly, they should be making more progress on each of these launches. This program is starting to look like it's slipping behind. \n"
    "You know, Starship was supposed to be able to at least orbit the Earth by now. And on this particular flight, the fact they couldn't hit reentry is a big problem because they wanted to test a lot of new experimental heat tiles. And they couldn't do that. So they're kind of further than they were on the last two flights. \n"
    "But they haven't made a lot of progress. And keep in mind the guy who owns SpaceX, Elon Musk, he has said that he hopes Starship will carry people to Mars one day, right? So do you think that's going to be happening anytime soon? You know, SpaceX has talked about sending a Starship without people to Mars as soon as next year. \n"
    "But I don't see how that happens. There's so much they need to work through to get the spacecraft working. In addition to the reentry problems, you know, they have to figure out how to refuel it in space to get the gas on to go to Mars. That's a really formidable challenge. NASA wants Starship to land people on the moon as soon as 2027. \n"
    "And even getting to the moon is starting to look like a pretty tough goal given how things are going. But you never count Elon Musk or SpaceX out. There's a lot of smart people working on this program. And you know, we'll just have to see what the next launch brings. Wait and see. That is Empire's Jeff Brumfield. Thank you so much, Jeff. Thank you, Elsa."
    )

    logger.info("[INFO] Running pipeline test with sample transcript...\n")
    report = initialize_pipeline(sample_transcript)

    print("\n📄 Final Generated Report:\n")
    print(report)