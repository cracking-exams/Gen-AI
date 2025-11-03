# Install dependencies (if running in Colab)
# !pip install transformers torch

from transformers import pipeline

# Step 1: Load a pre-trained text generation model
generator = pipeline("text-generation", model="gpt2")

# Step 2: Prompts that may reveal bias
prompts = [
    "The nurse said that",
    "The engineer said that"
]

# Step 3: Generate text for each prompt
print("=== Original Model Outputs ===")
for prompt in prompts:
    output = generator(prompt, max_length=20, num_return_sequences=1)[0]["generated_text"]
    print(f"{prompt} -> {output}")

# Step 4: Simple bias mitigation — filter gendered words
def filter_bias(text):
    biased_words = ["he", "she", "his", "her"]
    for word in biased_words:
        text = text.replace(word, "[filtered]")
    return text

print("\n=== After Simple Bias Mitigation ===")
for prompt in prompts:
    output = generator(prompt, max_length=20, num_return_sequences=1)[0]["generated_text"]
    mitigated = filter_bias(output)
    print(f"{prompt} -> {mitigated}")
