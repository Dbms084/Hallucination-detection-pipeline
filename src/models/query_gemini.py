import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

genai.configure(api_key=api_key)

model = genai.GenerativeModel("gemini-2.5-flash")


def ask_gemini(question):
    response = model.generate_content(question)
    return response.text

if __name__ == "__main__":
    question = "Who invented Python?"

    answer = ask_gemini(question)

    print(answer)