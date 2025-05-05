from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AdamW
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)
import torch
from tqdm import tqdm
import json

class T5FeedbackDataset(Dataset):
    def __init__(self, data, tokenizer, max_input_length=512, max_target_length=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        instance = self.data[idx]

        input_text = (
            f"The error type is: {instance['error_tag']}\n"
            f"The part that may contain the error is: \"{instance['error_phrase']}\"\n"
            f"The suggested correction is: \"{instance['correction']}\"\n"
            f"The original sentence is: {instance['source']}\"\n"
            f"The corrected sentence is: {instance['corrected']}"
        )
        target_text = instance["ref"]

        input_enc = self.tokenizer(
            input_text,
            truncation=True,
            padding="max_length",
            max_length=self.max_input_length,
            return_tensors="pt"
        )

        target_enc = self.tokenizer(
            target_text,
            truncation=True,
            padding="max_length",
            max_length=self.max_target_length,
            return_tensors="pt"
        )

        return {
            "input_ids": input_enc["input_ids"].squeeze(0),
            "attention_mask": input_enc["attention_mask"].squeeze(0),
            "labels": target_enc["input_ids"].squeeze(0)
        }
    
# Step 1: Load Tokenizer and Model
def load_model_and_tokenizer(model_name="t5-small"):
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    return tokenizer, model

# Step 2: Load and Preprocess Dataset (JSON version)
def preprocess_function(examples, tokenizer, max_input_length=512, max_target_length=128):
    inputs = tokenizer(
        examples["input_text"],
        max_length=max_input_length,
        padding="max_length",
        truncation=True,
    )
    targets = tokenizer(
        examples["target_text"],
        max_length=max_target_length,
        padding="max_length",
        truncation=True,
    )
    inputs["labels"] = targets["input_ids"]
    return inputs

def load_and_preprocess_json_dataset(train_path, valid_path, tokenizer):
    dataset = load_dataset("json", data_files={"train": train_path, "validation": valid_path})
    tokenized = dataset.map(lambda x: preprocess_function(x, tokenizer), batched=True)
    return tokenized

# Step 3: Training Arguments
def get_training_args(output_dir="./t5-feedback"):
    return TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=15,
        weight_decay=0.01,
        save_strategy="epoch",
        save_total_limit=2,
        predict_with_generate=True,
        logging_dir='./logs',
        logging_steps=10,
    )

# Step 4: Train the Model
def train_model(model, tokenizer, tokenized_datasets, training_args):
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    trainer.train()
    return trainer

def evaluate_dev(model, dev_data, tokenizer, device, batch_size=4):
    model.eval()
    model.to(device)

    dataset = T5FeedbackDataset(data=dev_data, tokenizer=tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size)

    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating dev loss"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # T5 expects -100 for ignored labels (not just padding tokens)
            labels[labels == tokenizer.pad_token_id] = -100

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss
            total_loss += loss.item()
            num_batches += 1

    avg_loss = total_loss / num_batches
    return avg_loss


# Step 5: Generate Feedback
def generate_feedback(model, tokenizer, input_text, max_length=128):
    model.eval()
    input_ids = tokenizer(input_text, return_tensors="pt", truncation=True).input_ids
    with torch.no_grad():
        outputs = model.generate(input_ids, max_length=max_length)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def inference(model, tokenizer, test_data, device):
    model.eval()
    model.to(device)

    output_data = []

    for instance in tqdm(test_data, desc="Generating test feedback"):
        # Unified input format (same for all instances)
        input_text = (
            "Explain the correction:\n"
            f"Error Tag: {instance['error_tag']}\n"
            f"Error Phrase: \"{instance['error_phrase']}\"\n"
            f"Correction: \"{instance['correction']}\"\n"
            f"Source Sentence: {instance['source']}"
        )

        encoding = tokenizer(
            input_text,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(device)

        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=encoding["input_ids"],
                attention_mask=encoding["attention_mask"],
                max_length=100,
                num_beams=4,
                early_stopping=True
            )

        generated_feedback = tokenizer.decode(
            generated_ids[0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )

        # Update output
        updated_instance = dict(instance)
        updated_instance["generated_feedback"] = generated_feedback
        output_data.append(updated_instance)

    return output_data


# === Load Data ===
with open("PATH TO YOUR TRAIN DATA") as f:
    train_data = json.load(f)
with open("PATH TO YOUR DEV DATA") as f:
    dev_data = json.load(f)

tokenizer = T5Tokenizer.from_pretrained("t5-small")
train_dataset = T5FeedbackDataset(train_data, tokenizer)
dev_dataset = T5FeedbackDataset(dev_data, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
dev_loader = DataLoader(dev_dataset, batch_size=4)

# === Model & Optimizer ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = T5ForConditionalGeneration.from_pretrained("t5-small").to(device)
optimizer = AdamW(model.parameters(), lr=5e-5)

# === Training Loop with Dev Evaluation ===
best_dev_loss = float("inf")
epochs = 15

for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        labels[labels == tokenizer.pad_token_id] = -100

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_train_loss = total_loss / len(train_loader)
    avg_dev_loss = evaluate_dev(model, dev_data, tokenizer, device)

    print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f}")
    print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Dev Loss: {avg_dev_loss:.4f}")

    # Save only the best model
    if avg_dev_loss < best_dev_loss:
        best_dev_loss = avg_dev_loss
        model.save_pretrained("t5_ref_feedback_best")
        tokenizer.save_pretrained("t5_ref_feedback_best")
        print("Saved new best model.")

# === Show First 3 Dev Predictions After Final Epoch ===
model.eval()
print("\nSample Dev Predictions (after final epoch):")
for i, instance in enumerate(dev_data[:3]):
    input_text = (
        "Explain the correction:\n"
        f"Error Tag: {instance['error_tag']}\n"
        f"Error Phrase: \"{instance['error_phrase']}\"\n"
        f"Correction: \"{instance['correction']}\"\n"
        f"Source Sentence: {instance['source']}"
    )

    encoding = tokenizer(input_text, return_tensors="pt", truncation=True).to(device)
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=encoding["input_ids"],
            attention_mask=encoding["attention_mask"],
            max_length=100,
            num_beams=4,
            early_stopping=True
        )
    generated_text = tokenizer.decode(
        generated_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True
    )

    print(f"\nExample {i+1}")
    print(f"SOURCE: {instance['source']}")
    print(f"CORRECTED: {instance['corrected']}")
    print(f"GOLD FB: {instance['ref']}")
    print(f"GEN FB: {generated_text}")

# === Load Test Data ===
with open("PATH TO YOUR TEST DATA") as f:
    test_data = json.load(f)

# === Generate Feedback for Test Data ===
print("\nGenerating feedback for test set...")
test_predictions = inference(model, tokenizer, test_data, device)

# === Save the Test Predictions into a JSON File ===
output_path = "OUTPUT PATH"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(test_predictions, f, ensure_ascii=False, indent=2)

print(f"\nTest set predictions saved to: {output_path}")
