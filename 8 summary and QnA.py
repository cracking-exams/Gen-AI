# Install dependencies
!pip install transformers datasets evaluate rouge_score sacrebleu -q

from transformers import pipeline
import evaluate

# --- 1. Summarization Task ---
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

text = """Artificial intelligence is transforming industries worldwide.
It automates repetitive tasks, enhances decision-making, and 
unlocks new opportunities in data-driven innovation."""

# Prompt (design)
prompt = f"Summarize this article: {text}"

# Generate summary
summary = summarizer(prompt, max_length=40, min_length=10, do_sample=False)[0]['summary_text']
print("Generated Summary:\n", summary)

# Reference for evaluation
reference = ["AI is revolutionizing industries by automating tasks and enabling data-driven decisions."]

# Evaluate with ROUGE
rouge = evaluate.load("rouge")
rouge_result = rouge.compute(predictions=[summary], references=reference)
print("\nROUGE Scores:", rouge_result)


# --- 2. Question Answering Task ---
qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

context = """Python is a programming language created by Guido van Rossum in 1991.
It is widely used for data science, AI, web development, and automation."""

question = "Who created Python?"

# Prompt (design)
qa_result = qa_pipeline(question=question, context=context)
print("\nQuestion:", question)
print("Answer:", qa_result['answer'])

# Evaluate using BLEU (for QA)
bleu = evaluate.load("sacrebleu")
reference_answer = [["Guido van Rossum"]]
bleu_result = bleu.compute(predictions=[qa_result['answer']], references=reference_answer)
print("\nBLEU Score:", bleu_result)
