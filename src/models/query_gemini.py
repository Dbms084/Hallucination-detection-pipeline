import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

genai.configure(api_key=api_key)

model = genai.GenerativeModel("gemini-2.5-flash")


from google.api_core.exceptions import ResourceExhausted
import time

def ask_gemini(question):
    max_retries = 5
    base_delay = 15
    for attempt in range(max_retries):
        try:
            response = model.generate_content(question)
            return response.text
        except (ResourceExhausted, Exception) as e:
            is_rate_limit = isinstance(e, ResourceExhausted) or "429" in str(e) or "quota" in str(e).lower()
            if is_rate_limit and attempt < max_retries - 1:
                is_daily_limit = "daily" in str(e).lower() or "limit: 20" in str(e) or "quota_id" in str(e).lower() or "RequestsPerDay" in str(e)
                if is_daily_limit:
                    ollama_model = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
                    print(f"\n[WARNING] Gemini Query hit daily API quota limit. Falling back to local Ollama model ({ollama_model})...")
                    try:
                        import json
                        import urllib.request
                        url = "http://localhost:11434/api/generate"
                        data = {"model": ollama_model, "prompt": question, "stream": False}
                        req = urllib.request.Request(
                            url,
                            data=json.dumps(data).encode("utf-8"),
                            headers={"Content-Type": "application/json"}
                        )
                        with urllib.request.urlopen(req, timeout=120) as resp:
                            resp_data = json.loads(resp.read().decode("utf-8"))
                            return resp_data.get("response", "")
                    except Exception as ollama_err:
                        print(f"[ERROR] Local Ollama fallback query also failed: {ollama_err}")
                        raise e
                print(f"\n[WARNING] Gemini Query hit rate limit. Sleeping {base_delay} seconds before retry (Attempt {attempt+1}/{max_retries})...")
                time.sleep(base_delay)
            else:
                is_api_error = isinstance(e, ResourceExhausted) or "429" in str(e) or "quota" in str(e).lower()
                if is_api_error:
                    ollama_model = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
                    print(f"\n[WARNING] Gemini Query failed after retries or quota limits. Falling back to local Ollama model ({ollama_model})...")
                    try:
                        import json
                        import urllib.request
                        url = "http://localhost:11434/api/generate"
                        data = {"model": ollama_model, "prompt": question, "stream": False}
                        req = urllib.request.Request(
                            url,
                            data=json.dumps(data).encode("utf-8"),
                            headers={"Content-Type": "application/json"}
                        )
                        with urllib.request.urlopen(req, timeout=120) as resp:
                            resp_data = json.loads(resp.read().decode("utf-8"))
                            return resp_data.get("response", "")
                    except Exception as ollama_err:
                        print(f"[ERROR] Local Ollama fallback query also failed: {ollama_err}")
                        raise e
                raise e

if __name__ == "__main__":
    question = "Who invented Python?"

    answer = ask_gemini(question)

    print(answer)