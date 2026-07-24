import os
import google.generativeai as genai
from dotenv import load_dotenv

# Load your API key from the .env file
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

print("Fetching available models for your API key...\n")
for m in genai.list_models():
    if 'generateContent' in m.supported_generation_methods:
        print(m.name.replace("models/", ""))