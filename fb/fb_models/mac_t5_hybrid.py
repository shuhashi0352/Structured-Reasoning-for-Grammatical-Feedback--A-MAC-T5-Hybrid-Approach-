import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.optim import AdamW
import torch.nn as nn
from tqdm import tqdm
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput
import json
import random


"""""
JSON Instance Parser (Serialization)
"""""

#### FOR Out-of-Distribution (OOD) -- shuffling the structure

def serialize_input(json_instance, step):
    """
    Serialize input fields for a given MAC step.
    
    Args:
        json_instance (dict): One training example.
        step (int): MAC step index (1–4).
    
    Returns:
        str: Serialized and shuffled input string.
    """
    # === Extract all fields ===
    error_tag = json_instance.get("error_tag", "")
    error_phrase = json_instance.get("error_phrase", "")
    correction = json_instance.get("correction", "")
    source = json_instance.get("source", "")
    corrected = json_instance.get("corrected", "")
    feedback = json_instance.get("feedback", "")
    feedback_steps = feedback.strip().split("\n")

    reasoning = ""
    if feedback:
        feedback_steps = feedback.strip().split("\n")
        if 1 <= step <= 4 and len(feedback_steps) >= step:
            reasoning = feedback_steps[step - 1]

    # === Apply step-based masking ===

    # Step 1: Identify where the error is
    # Focus on the sentence context and the type/location of the error
    # Mask fix-related fields (`correction`, `corrected`) and `feedback`
    if step == 1:
        chunks = [
            f"The error type is: {error_tag}.",
            f"The original sentence is: {source}",
            f"The part that may contain the error is: \"{error_phrase}\"."
            f"This indicates that the sentence involves a grammatical issue related to: {reasoning}."
        ]

    # Step 2: Figure out what the fix is
    # Focus on identifying the correction and its relationship to the error phrase
    # Mask source context and `feedback`
    elif step == 2:
        chunks = [
            f"The part that may contain the error is: \"{error_phrase}\".",
            f"The suggested correction is: \"{correction}\".",
            f"The corrected sentence is: {corrected}"
            f"This part is incorrect because: {reasoning}."
        ]

    # Step 3: Understand the rule that was violated
    # Review everything except the final explanation
    # Mask only the `feedback`
    elif step == 3:
        chunks = [
            f"The error type is: {error_tag}.",
            f"The part that may contain the error is: \"{error_phrase}\".",
            f"The original sentence is: {source}",
            f"The corrected sentence is: {corrected}.",
            f"The grammatical rule that was broken is: {reasoning}."
        ]

    # Step 4: Explain why the correction works (Option 1)
    # No masking — give the model full access to all fields including the CoT `feedback`
    elif step == 4:
        # Option 1: No masking — include everything
        chunks = [
            f"The error type is: {error_tag}.",
            f"The part that may contain the error is: \"{error_phrase}\".",
            f"The original sentence is: {source}",
            f"The corrected sentence is: {corrected}.",
            f"The correction made is: \"{correction}\".",
            f"This correction works because: {reasoning}."
        ]
    else:
        raise ValueError(f"Invalid step {step}. Expected 1–4.")

    # === Shuffle the fields ===
    random.shuffle(chunks)

    # === Combine into one string ===
    structured_input = " ".join(chunks)
    return structured_input

#### FOR In-Distribution
# def serialize_input(json_instance):
#     # Input fields
#     error_tag = json_instance["error_tag"]
#     error_phrase = json_instance["error_phrase"]
#     correction = json_instance["correction"]
#     source = json_instance["source"]
#     corrected = json_instance["corrected"]

#     # Structure the input text field-wise
#     structured_input = f"<error_tag> {error_tag} </error_tag> " \
#                        f"<error_phrase> {error_phrase} </error_phrase> " \
#                        f"<correction> {correction} </correction> " \
#                        f"<source> {source} </source> " \
#                        f"<corrected> {corrected} </corrected>"
#     return structured_input


class GECFeedbackDataset(Dataset):
    def __init__(self, data, tokenizer, step, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.step = step
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        instance = self.data[idx]
        input_text = serialize_input(instance, step=self.step)
        target_text = instance["ref"]

        enc = self.tokenizer(
            input_text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        dec = self.tokenizer(
            target_text,
            truncation=True,
            padding="max_length",
            max_length=150,
            return_tensors="pt"
        )

        return {
            "input_ids": enc.input_ids.squeeze(),
            "attention_mask": enc.attention_mask.squeeze(),
            "labels": dec.input_ids.squeeze()  # this is what loss will use
        }
    

"""""
MAC Reasoning Architecture
"""""

class MACReasoning(nn.Module):
    def __init__(self, dim, num_steps=4):
        super().__init__()
        self.num_steps = num_steps
        self.control_proj = nn.Linear(dim * 2, dim)
        self.read_proj = nn.Linear(dim, dim)
        self.write_proj = nn.Linear(dim * 2, dim)

    """
    batch_size = 1
    """

    # def forward(self, context_vectors):
    #     memory = torch.zeros(context_vectors.shape[1]).to(context_vectors.device)
    #     control = torch.zeros_like(memory)

    #     for _ in range(self.num_steps):
    #         control_input = torch.cat([torch.mean(context_vectors, dim=0), memory], dim=-1)
    #         control = torch.tanh(self.control_proj(control_input))
    #         attn_weights = torch.softmax(torch.matmul(context_vectors, control), dim=0)
    #         read = torch.sum(attn_weights.unsqueeze(-1) * context_vectors, dim=0)
    #         combined = torch.cat([memory, read], dim=-1)
    #         memory = torch.tanh(self.write_proj(combined))

    #     return memory

    """
    batch_size > 1
    """

    def forward(self, context_vectors):
        """
        context_vectors: [batch_size, seq_len, dim]
        """
        batch_size, seq_len, dim = context_vectors.size()
    
        memory = torch.zeros(batch_size, dim).to(context_vectors.device)
        control = torch.zeros_like(memory)

        for _ in range(self.num_steps):
            mean_context = torch.mean(context_vectors, dim=1)  # [batch_size, dim]
            control_input = torch.cat([mean_context, memory], dim=-1)  # [batch_size, 2*dim]
            control = torch.tanh(self.control_proj(control_input))     # [batch_size, dim]

            # Attention weights: [batch_size, seq_len]
            attn_weights = torch.softmax(torch.einsum("bsd,bd->bs", context_vectors, control), dim=1)

            # Weighted sum: [batch_size, dim]
            read = torch.einsum("bsd,bs->bd", context_vectors, attn_weights)

            combined = torch.cat([memory, read], dim=-1)  # [batch_size, 2*dim]
            memory = torch.tanh(self.write_proj(combined))  # [batch_size, dim]

        return memory  # shape: [batch_size, dim]


"""
T5 Encoder-Decoder Architecture
"""

class MACWithT5(nn.Module):
    def __init__(self, t5_model, mac_module):
        super().__init__()
        self.t5 = t5_model
        self.mac = mac_module

    def forward(self, input_ids, attention_mask, labels=None):
        enc_out = self.t5.encoder(input_ids=input_ids, attention_mask=attention_mask)
        context_vectors = enc_out.last_hidden_state.squeeze(0)  # [seq_len, dim]

        memory_vector = self.mac(context_vectors)

        # # batch_size = 1
        # extended = torch.cat([
        #     memory_vector.unsqueeze(0).unsqueeze(0),
        #     enc_out.last_hidden_state
        # ], dim=1)

        # batch_size > 1
        extended = torch.cat([
            memory_vector.unsqueeze(1),  # [batch_size, 1, dim]
            enc_out.last_hidden_state    # [batch_size, seq_len, dim]
        ], dim=1)

        decoder_input_ids = self.t5._shift_right(labels)

        outputs = self.t5(
            inputs_embeds=None,
            encoder_outputs=(extended,),
            decoder_input_ids=decoder_input_ids,
            labels=labels
        )

        return outputs.loss


"""
Training
"""

def train(model, data, tokenizer, optimizer, device, epochs=15, batch_size=4):
    #  Trains the MAC-T5 model over all MAC steps (1–4) for multiple epochs.

    model.train()
    model.to(device)

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}")
        
        for step in range(1, 5):  # MAC steps 1 to 4
            print(f"Step {step}")

            # Create dataset with step-specific masking
            dataset = GECFeedbackDataset(data=data, tokenizer=tokenizer, step=step)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

            model.train()
            total_loss = 0.0
            for batch in tqdm(dataloader, desc=f"Step {step}"):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                loss = model(input_ids, attention_mask, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(dataloader)
            print(f"Step {step} | Avg Loss: {avg_loss:.4f}")

            dev_loss = evaluate_dev(model, dev_data, tokenizer, device, step, batch_size)
            print(f"Step {step} | Dev Loss: {dev_loss:.4f}")


"""
Evaluation
"""

def evaluate_dev(model, dev_data, tokenizer, device, step, batch_size=4):
    model.eval()
    dataset = GECFeedbackDataset(data=dev_data, tokenizer=tokenizer, step=step)
    dataloader = DataLoader(dataset, batch_size=batch_size)

    total_loss = 0.0
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            loss = model(input_ids, attention_mask, labels)
            total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    return avg_loss



def inference(model, tokenizer, test_data, device):
    model.eval()
    model.to(device)

    output_data = []

    # Run reasoning steps 1 to 3
    for instance in tqdm(test_data, desc="Generating test feedback"):
        # Step 1–3: encode reasoning
        for step in range(1, 4):
            input_text = serialize_input(instance, step)
            encoding = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True).to(device)

            with torch.no_grad():
                encoder_output = model.t5.encoder(**encoding)

                # batch_size = 1
                # context_vectors = encoder_output.last_hidden_state.squeeze(0)

                # bacth_size > 1
                context_vectors = encoder_output.last_hidden_state 

                memory_vector = model.mac(context_vectors)

        # Step 4: generate feedback
        extended = torch.cat([
            memory_vector.unsqueeze(0),
            encoder_output.last_hidden_state
        ], dim=1)

        extended_attention_mask = torch.cat([
            torch.ones((1, 1), dtype=encoding["attention_mask"].dtype).to(device),  # allow attending to memory
            encoding["attention_mask"]
        ], dim=1)

        encoder_output = BaseModelOutput(last_hidden_state=extended)

        with torch.no_grad():
            generated_ids = model.t5.generate(
                encoder_outputs=encoder_output,
                attention_mask=extended_attention_mask,
                max_length=100,
                num_beams=4,
                early_stopping=True
            )

        generated_feedback = tokenizer.decode(generated_ids[0], skip_special_tokens=True,  clean_up_tokenization_spaces=True)

        # Update and collect the instance
        updated_instance = dict(instance)
        updated_instance["generated_feedback"] = generated_feedback
        output_data.append(updated_instance)

    return output_data




# Initialize tokenizer and model
tokenizer = T5Tokenizer.from_pretrained("t5-small")
t5_model = T5ForConditionalGeneration.from_pretrained("t5-small")

# If you have the save point after the pretraining process
# tokenizer = T5Tokenizer.from_pretrained("t5_ref_feedback_pretrained")
# t5_model = T5ForConditionalGeneration.from_pretrained("t5_ref_feedback_pretrained")

mac_module = MACReasoning(dim=512, num_steps=4)  # 512 = t5-base hidden size

# Wrap the full model
model = MACWithT5(t5_model, mac_module)
optimizer = AdamW(model.parameters(), lr=5e-5)

# Load json files
with open("PATH TO YOUR TRAIN DATA") as f:
    train_data = json.load(f)

with open("PATH TO YOUR DEV DATA") as f:
    dev_data = json.load(f)

device = "cuda" if torch.cuda.is_available() else "cpu"

# Train
train(model, train_data, tokenizer, optimizer, device, epochs=15)

torch.save(model.state_dict(), "mac_t5_checkpoint.pt")
print("Model saved.")


# Development
dev_results = inference(model, tokenizer, dev_data, device)

with open("mac_t5_generated_dev_feedback.json", "w") as f:
    json.dump(dev_results, f, indent=2, ensure_ascii=False)

print("Dev feedback generated and saved.")

for i in range(3):  # Show first 3 examples
    print(f"\nExample {i+1}")
    print("SOURCE:   ", dev_results[i]["source"])
    print("CORRECTED:", dev_results[i]["corrected"])
    print("GOLD FB:  ", dev_results[i]["ref"])
    print("GEN  FB:  ", dev_results[i]["generated_feedback"])
    
# Test
with open("PATH TO YOUR TEST DATA") as f:
    test_data = json.load(f)

results = inference(model, tokenizer, test_data, device)

with open("oneshot_mac_t5_generated_test_feedback.json", "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print("Generated feedback saved to mac_t5_generated_test_feedback.json.")




# Function for attention heatmaps (for the future work)
def visualize_training_attention(model, tokenizer, data_path, step=1, num_examples=5, device="cpu"):
    """
    Visualize cross-attention heatmaps for examples using the actual training prompt logic.

    Args:
        model: The MACWithT5 model.
        tokenizer: Tokenizer used for T5.
        data_path (str): Path to JSON file containing training data.
        step (int): MAC step to use (1–4).
        num_examples (int): Number of examples to visualize.
        device (str): "cpu" or "cuda".
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import torch
    import json
    import random

    # === Helper: replicate serialize_input from training ===
    def serialize_input(json_instance, step):
        error_tag = json_instance.get("error_tag", "")
        error_phrase = json_instance.get("error_phrase", "")
        correction = json_instance.get("correction", "")
        source = json_instance.get("source", "")
        corrected = json_instance.get("corrected", "")
        feedback = json_instance.get("feedback", "")
        feedback_steps = feedback.strip().split("\n")

        reasoning = ""
        if feedback:
            if 1 <= step <= 4 and len(feedback_steps) >= step:
                reasoning = feedback_steps[step - 1]

        if step == 1:
            chunks = [
                f"The error type is: {error_tag}.",
                f"The original sentence is: {source}",
                f"The part that may contain the error is: \"{error_phrase}\"."
                f"This indicates that the sentence involves a grammatical issue related to: {reasoning}."
            ]
        elif step == 2:
            chunks = [
                f"The part that may contain the error is: \"{error_phrase}\".",
                f"The suggested correction is: \"{correction}\".",
                f"The corrected sentence is: {corrected}"
                f"This part is incorrect because: {reasoning}."
            ]
        elif step == 3:
            chunks = [
                f"The error type is: {error_tag}.",
                f"The part that may contain the error is: \"{error_phrase}\".",
                f"The original sentence is: {source}",
                f"The corrected sentence is: {corrected}.",
                f"The grammatical rule that was broken is: {reasoning}."
            ]
        elif step == 4:
            chunks = [
                f"The error type is: {error_tag}.",
                f"The part that may contain the error is: \"{error_phrase}\".",
                f"The original sentence is: {source}",
                f"The corrected sentence is: {corrected}.",
                f"The correction made is: \"{correction}\".",
                f"This correction works because: {reasoning}."
            ]
        else:
            raise ValueError(f"Invalid step {step}. Expected 1–4.")

        random.shuffle(chunks)
        return " ".join(chunks)

    # === Load training data ===
    with open(data_path, "r") as f:
        data = json.load(f)

    model.eval()
    model.to(device)

    for i, instance in enumerate(data[:num_examples]):
        input_text = serialize_input(instance, step=step)
        target_text = instance["ref"]

        inputs = tokenizer(input_text, return_tensors="pt", truncation=True).to(device)
        outputs = tokenizer(target_text, return_tensors="pt", truncation=True).to(device)

        with torch.no_grad():
            result = model.t5(
                input_ids=inputs["input_ids"],
                decoder_input_ids=outputs["input_ids"],
                output_attentions=True,
                return_dict=True
            )

        cross_att = result.cross_attentions[-1][0]
        avg_att = cross_att.mean(dim=0)

        input_tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        output_tokens = tokenizer.convert_ids_to_tokens(outputs["input_ids"][0])

        plt.figure(figsize=(len(input_tokens) * 0.5, len(output_tokens) * 0.5))
        sns.heatmap(avg_att.cpu(), xticklabels=input_tokens, yticklabels=output_tokens, cmap="viridis")
        plt.xlabel("Input Tokens")
        plt.ylabel("Output Tokens")
        plt.title(f"Step {step} – Example {i+1}")
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()