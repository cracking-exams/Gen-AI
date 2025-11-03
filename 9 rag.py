# Install dependencies first (uncomment if running on Colab)
# !pip install faiss-cpu transformers torch sentence-transformers

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

# Step 1: Prepare small knowledge base
docs = [
    "Python is a high-level programming language used for AI and data science.",
    "FAISS is a library for efficient similarity search and clustering of dense vectors.",
    "Transformers are deep learning models used in NLP tasks like translation and summarization.",
    "Retrieval-Augmented Generation combines information retrieval with language generation.",
]

# Step 2: Create embeddings using SentenceTransformer
embedder = SentenceTransformer('all-MiniLM-L6-v2')
doc_embeddings = embedder.encode(docs, convert_to_numpy=True)

# Step 3: Create FAISS index and add embeddings
dimension = doc_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(doc_embeddings)

# Step 4: Query and retrieve relevant docs
query = "How does RAG work in NLP?"
query_embedding = embedder.encode([query])
_, indices = index.search(query_embedding, k=2)

retrieved_docs = [docs[i] for i in indices[0]]
context = " ".join(retrieved_docs)

# Step 5: Generate answer using small LLM
tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
model = AutoModelForCausalLM.from_pretrained("distilgpt2")

prompt = f"Context: {context}\nQuestion: {query}\nAnswer:"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_length=100, do_sample=True, top_p=0.9)

print("=== Retrieved Context ===")
print(context)
print("\n=== Generated Answer ===")
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
