import openai 
import json
import time

# === Set your OpenAI API Key ===
openai.api_key = "PASTE YOUR API KEY HERE"

# === Load both outputs ===
with open("PATH TO MAC-T5 GENERATED FEEDBACK") as f:
    transformer_outputs = json.load(f)

with open("PATH TO T5 GENERATED FEEDBACK") as f:
    mac_t5_outputs = json.load(f)

assert len(transformer_outputs) == len(mac_t5_outputs), "Mismatch between the number of instances"

# === Evaluate and Collect Results ===
results = []

for idx, (t_instance, m_instance) in enumerate(zip(transformer_outputs, mac_t5_outputs)):
    prompt = f"""You are evaluating two different pieces of feedback that correct a student's English sentence.

Each feedback aims to explain:
- What is wrong
- Why it is wrong
- And help the student learn the grammar better.

Here is the information for the case:
- Error Tag: {t_instance['error_tag']}
- Error Phrase: {t_instance['error_phrase']}
- Correction: {t_instance['correction']}
- Source Sentence: {t_instance['source']}
- Corrected Sentence: {t_instance['corrected']}
- Reference Explanation: {t_instance['ref']}

Now, here are two feedback explanations to compare:

Feedback A (Transformer-based):
{t_instance['generated_feedback']}

Feedback B (MAC-T5-based):
{m_instance['generated_feedback']}

According to the following rubric:
"Explains not only what is wrong, but also why it is wrong, making feedback more educational and learner-friendly."

Which feedback is better? Answer only with one word: "A" or "B".
"""

    # === Call OpenAI API ===
    try:
        client = openai.OpenAI(api_key=openai.api_key)
        response = client.chat.completions.create(
            model="gpt-4",  # or "gpt-3.5-turbo" if you want cheaper/faster
            messages=[
                {"role": "system", "content": "You are a strict and fair English feedback evaluator."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            max_tokens=5
        )
        evaluation = response.choices[0].message.content.strip()

        results.append({
            "index": idx,
            "evaluation": evaluation,
            "feedback_a": t_instance['generated_feedback'],
            "feedback_b": m_instance['generated_feedback']
        })

        print(f"Instance {idx+1}: {evaluation}")

        time.sleep(1)  # polite waiting to avoid rate limit

    except Exception as e:
        print(f"Error at instance {idx+1}: {e}")
        time.sleep(5)  # wait longer if error

# === Save the Evaluation Results ===
with open("twoshot_feedback_comparison_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print("\nFinished evaluating all instances!")