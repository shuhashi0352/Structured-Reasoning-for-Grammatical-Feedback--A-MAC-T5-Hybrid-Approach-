import openai
import json
import time
import random

# Your OpenAI API key here
openai.api_key = "PASTE YOUR API KEY HERE"

# Load your input JSON file
with open("PATH TO ANY TRAIN/DEV/TEST DATA", "r") as f:
    data = json.load(f)

output_data = []

# Loop over each entry in the JSON
for i, item in enumerate(data):
    # Build the structured prompt
    prompt = f"""
You are a grammar feedback assistant. Your task is to write a short but clear sentence that explains what was grammatically wrong in the learner's sentence and how to fix it.

Your explanation must follow these rules:

-  **Explain the general grammar rule** that was broken.
-  **Do NOT just point out the error or repeat the correction.**
-  Make the feedback useful for learners so they can apply the rule in other situations.
-  Use clear and simple language — avoid technical jargon if possible.
-  Keep your response under 80 words.
-  Just focus on the part of the sentence that the "error_phrase" and "correction" refer to. **Do NOT mention other errors** that are not part of the specified correction.

Here is an example of high-quality feedback that follows these guidelines:

Example:
  {{
    "error_tag": "U:DET",
    "error_phrase": "the",
    "correction": "",
    "source": "having the fitness classes is a very critical thing...",
    "corrected": "having fitness classes is a very critical thing..."
  }}

Feedback: "The error in your sentence is the unnecessary use of the article \"the\" before the plural noun phrase \"fitness classes.\" In English, plural or non-specific nouns used in a general sense do not typically require a definite article. You should say \"having fitness classes,\" not \"having the fitness classes.\""

Now follow the same format to give feedback for the input below:

Input:
  {{
    "error_tag": "{item['error_tag']}",
    "error_phrase": "{item['error_phrase']}",
    "correction": "{item['correction']}",
    "source": "{item['source']}",
    "corrected": "{item['corrected']}"
  }}
"""


    try:
        client = openai.OpenAI(api_key = "PASTE YOUR API KEY HERE")  # Initialize client at the top of your script

        response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4,
        )

        feedback = response.choices[0].message.content.strip()
        item["feedback"] = feedback
        output_data.append(item)

        print(f"[{i+1}]", item)
        time.sleep(1.2)  # Delay to respect rate limits

    except Exception as e:
        print(f"[{i+1}] Error: {e}")
        item["feedback"] = "ERROR"
        output_data.append(item)

# Save the results with feedback
with open("OUTPUT PATH", "w") as f:
    json.dump(output_data, f, indent=2, ensure_ascii=False)

print("All feedback generated and saved.")
