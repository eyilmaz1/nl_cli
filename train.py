from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    TrainingArguments,
    Trainer,
)

MODEL_NAME = "google/flan-t5-small"

# Load base model + tokenizer
print("Loading tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
print("Tokenizer and model loaded ✅\n")

# Load your JSONL dataset
print("Loading dataset...")
dataset = load_dataset("json", data_files="nl_to_cli.jsonl")
print(f"Dataset loaded: {dataset}\n")
print("Sample dataset entry:")
print(dataset["train"][0], "\n")

# Tokenization function
def tokenize(batch):
    nl_texts = batch["input"]
    
    # batch["output"] is a list of dicts
    cli_texts = [" && ".join(item["cmd"]) for item in batch["output"]]

    inputs = tokenizer(nl_texts, padding="max_length", truncation=True, max_length=128)
    labels = tokenizer(cli_texts, padding="max_length", truncation=True, max_length=128)

    inputs["labels"] = labels["input_ids"]
    return inputs

print("Tokenizing dataset...")
tokenized_data = dataset.map(
    tokenize,
    batched=True,
    remove_columns=["input", "action", "output"]
)
print("Tokenization complete ✅\n")
print("Sample tokenized entry:")
print({k: v[:10] for k, v in tokenized_data["train"][0].items()}, "\n")  # show first 10 tokens

collator = DataCollatorForSeq2Seq(tokenizer, model=model)

training_args = TrainingArguments(
    output_dir="./checkpoint",
    per_device_train_batch_size=4,
    learning_rate=5e-5,
    num_train_epochs=5,
    logging_steps=10,
    save_strategy="epoch",
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_data["train"],
    data_collator=collator,
)

# Train
print("Starting training...")
trainer.train()
print("Training complete ✅\n")

# Save final model
print("Saving model and tokenizer...")
trainer.save_model("cli_model")
tokenizer.save_pretrained("cli_model")
print("Model saved to ./cli_model ✅")
