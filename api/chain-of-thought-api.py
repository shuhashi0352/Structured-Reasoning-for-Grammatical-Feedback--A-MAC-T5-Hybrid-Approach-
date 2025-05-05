import openai
import json
import time

# Your OpenAI API key here
openai.api_key = "PASTE YOUR API KEY HERE"

# Load input and reference data
input_path = "PATH TO ANY TRAINING DATA"
ref_path = "PATH TO REFERENCE FEEDBACK FOR THE CORRESPOONDING TRAINING DATA"

with open(input_path, "r") as f:
    data = json.load(f)

with open(ref_path, "r") as f:
    reference_feedback = json.load(f)

reference_dict = {
    (item["error_tag"], item["error_phrase"], item["source"]): item.get("feedback", "")
    for item in reference_feedback
}

output_data = []

# Loop over each entry
for i, item in enumerate(data):
    key = (item["error_tag"], item["error_phrase"], item["source"])
    ref_feedback = reference_dict.get(key, "")

    prompt = f"""
You are a reasoning assistant helping generate step-by-step grammatical explanations for a grammar correction model.

Given the following structured input, output a 4-step explanation that describes:
1. What kind of grammatical issue is involved (based on the error tag)
2. What exactly is wrong in the sentence
3. Why it is wrong — explain the grammatical rule that was broken so the learner can avoid the mistake in the future. Do not say just that it's not appropriate or not correct.
4. How it is corrected

Important:
- Just focus on the corrected part. Do not mention errors that are not highlighted.
- Do not simply restate the correction. Ground your explanation in general grammar rules.
- Avoid technical jargon, but be clear and instructive.

Here is a sample feedback written by a teacher that explains the same correction:
\"\"\"{ref_feedback}\"\"\"

Input:
  {{
    "error_tag": "{item['error_tag']}",
    "error_phrase": "{item['error_phrase']}",
    "correction": "{item['correction']}",
    "source": "{item['source']}",
    "corrected": "{item['corrected']}"
  }}

Output:
"""

    try:
        client = openai.OpenAI(api_key=openai.api_key)

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

# Save the results
with open("OUTPUT PATH", "w") as f:
    json.dump(output_data, f, indent=2, ensure_ascii=False)

print("All feedback generated and saved.")