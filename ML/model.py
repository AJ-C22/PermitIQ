from dotenv import load_dotenv
import os
import google.generativeai as genai

load_dotenv()

genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))

def generate():
    model = genai.GenerativeModel("gemini-2.0-flash-thinking-exp-01-21")  

    prompt = "What's 2 + 2?"

    response = model.generate_content(prompt)
    print(response.text)

if __name__ == "__main__":
    generate()
