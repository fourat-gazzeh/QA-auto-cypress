import os
import json
import requests
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={API_KEY}"

def load_html(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

def build_prompt(html_content): 
    return f"""
You are an expert in frontend UI analysis. Given the following HTML content:

**TASK 1: Extract elements**
Return a list of all input and button elements with the following details:
- tag: input or button
- type: e.g., text, email, submit, etc.
- id
- name
- class
- placeholder or label (if any)
- required: true or false (based on 'required' attribute)
- behavior: short description of what this element does, if it has an action or intention (e.g., "Triggers a search", "Submits the form", "Navigates to next step", "Clears the input field")

**TASK 2: User scenarios**
Describe ONLY 2 short usage scenarios:
1. ✅ A correct scenario that shows expected use of the page (e.g., user fills all required inputs and submits).
2. ❌ A faulty scenario that could happen if a user forgets to fill a required field or misuses the buttons.

**Format your answer strictly in this JSON format:**

{{
  "elements": [
    {{
      "tag": "input" | "button",
      "type": "...",
      "id": "...",
      "name": "...",
      "class": "...",
      "placeholder": "...",
      "required": true | false,
      "behavior": "..."
    }},
    ...
  ],
  "scenarios": [
    "Correct scenario description...",
    "Incorrect scenario description..."
  ]
}}

HTML to analyze:
{html_content}
"""

def call_gemini(prompt):
    headers = {"Content-Type": "application/json"}
    body = {
        "contents": [
            {
                "parts": [{"text": prompt}]
            }
        ]
    }

    response = requests.post(GEMINI_URL, headers=headers, json=body)
    response.raise_for_status()
    content = response.json()
    
    output_text = content["candidates"][0]["content"]["parts"][0]["text"]
    return output_text.strip().replace("```json", "").replace("```", "")

def save_json(output_str, file_path="outputwiss2.json"):
    try:
        json_data = json.loads(output_str)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(json_data, f, indent=2)
        print(f"[✓] Saved Gemini response to {file_path}")
    except json.JSONDecodeError:
        print("[!] Couldn't decode JSON. Here's the raw response:")
        print(output_str)

def main():
    html = load_html("saved_content3.html")
    prompt = build_prompt(html)
    output_str = call_gemini(prompt)
    save_json(output_str)

if __name__ == "__main__":
    main()
