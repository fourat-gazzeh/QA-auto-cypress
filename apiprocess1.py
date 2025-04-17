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

def build_prompt(html_content, page_name="Untitled Page", url_path="/"):
    return f"""
You are an expert in frontend UI analysis. Given the following HTML content, your task is to:

**TASK 1: Extract elements**
Return a list of all input and button elements with the following details:
- tag: input or button
- lable
- type: e.g., text, email, submit, etc.
- id
- name
- class
- placeholder or label (if any)
- required: true or false (based on 'required' attribute)

**TASK 2: Page Description**
Provide a description of the page including what it does and what the user expects to interact with:
- Describe what the page is intended for (e.g., login page, registration form).
- Mention the main interactive elements (e.g., buttons, input fields).
- Provide insights on what happens when the user interacts with these elements (e.g., form submission, validation, reset).

**TASK 3: User scenarios**
Describe ONLY 2 short usage scenarios:
1. ✅ A correct scenario that shows expected use of the page (e.g., user fills all required inputs and submits).
2. ❌ A faulty scenario that could happen if a user forgets to fill a required field or misuses the buttons.

**Format your answer strictly in this JSON format:**

{{
  "page_info": {{
    "name": "{page_name}",
    "description": "Description of the page and its functionality.",
    "url_path": "{url_path}"
  }},
  "elements": [
    {{
      "tag": "input" | "button",
      "type": "...",
      "lable": "...",
      "id": "...",
      "name": "...",
      "class": "...",
      "placeholder": "...",
      "required": true | false
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

def save_json(output_str, file_path="outputdesc1.json"):
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
