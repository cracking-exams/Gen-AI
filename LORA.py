import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, TrainingArguments, Trainer
from datasets import Dataset
import pandas as pd
from peft import LoraConfig, get_peft_model, TaskType

# Load model and tokenizer
model_name = "distilgpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Add padding token if it doesn't exist
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Configure LoRA
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=8,  # rank
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["c_attn"]  # target the attention layers
)

# Apply LoRA to model
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Sample training data (replace with your corpus)
texts = [
    "The quick brown fox jumps over the lazy dog.",
    "Machine learning is a subset of artificial intelligence.",
    "Transformers have revolutionized natural language processing.",
    "Fine-tuning pre-trained models requires careful hyperparameter tuning.",
    "LoRA reduces the number of trainable parameters significantly."
] * 20  # Repeat to create more samples

# Create dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding=True, max_length=128)

dataset = Dataset.from_dict({"text": texts})
tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Training arguments
training_args = TrainingArguments(

    num_train_epochs=3,
    per_device_train_batch_size=4,
    save_steps=500,
    logging_steps=100,
    evaluation_strategy="no",
    learning_rate=5e-4,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

# Fine-tune with LoRA
trainer.train()

# Save the model
# model.save_pretrained("./lora-gpt2-final")

# Text generation function
def generate_text(prompt, max_length=50):
    inputs = tokenizer(prompt, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_length=max_length,
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Generate samples
prompts = [
    "The future of AI",
]

print("Generated Text Samples:")
for prompt in prompts:
    generated = generate_text(prompt)
    print(f"Prompt: {prompt}")
    print(f"Generated: {generated}")
    print("-" * 50)

# Perplexity evaluation
def calculate_perplexity(text):
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs.input_ids)
        loss = outputs.loss
        perplexity = torch.exp(loss)
    return perplexity.item()

# Evaluate perplexity on sample texts
test_texts = [
    "The quick brown fox jumps over the lazy dog.",
    "Machine learning models require large amounts of data for training."
]

print("\nPerplexity Evaluation:")
for text in test_texts:
    perplexity = calculate_perplexity(text)
    print(f"Text: {text}")
    print(f"Perplexity: {perplexity:.2f}")
    print("-" * 50)