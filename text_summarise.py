import os
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_groq import ChatGroq


# ====== 1. Set your Groq API Key ======
load_dotenv()

# Get the API key from environment variable
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
print("🔑 Loaded key:", os.getenv("GROQ_API_KEY"))

if not GROQ_API_KEY:
    raise ValueError("❌ No GROQ_API_KEY found! Please set it in your .env file.")

print("✅ GROQ_API_KEY loaded successfully.")
llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name="llama-3.1-8b-instant",  # or use "llama3-8b-8192"
    temperature=0
)

# Load transcript
with open("daily_scrum_transcript_plain.txt", "r", encoding="utf-8") as f:
    transcript = f.read()


# Prompt for role-based summary
template = """
You are an AI meeting assistant. Summarize the following Scrum meeting 
from the perspective of the {role} team. 

Include:
- Key discussion points
- Action items
- Decisions made
- Issues or blockers
- Special focus on what is most relevant for {role} team.

Transcript:
{transcript}
"""
prompt = PromptTemplate(input_variables=["role", "transcript"], template=template)



chain = LLMChain(llm=llm, prompt=prompt)

# Example usage
result = chain.run(role="QA", transcript=transcript)
print("\n==== MEETING SUMMARY FOR QA TEAM ====\n")
print(result)

