from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")

# imagine these are chunks from a document
documents = [
    "Python is the best language for AI development",
    "Kathmandu is the capital city of Nepal",
    "LLMs are large language models trained on text",
    "The Himalayas are the tallest mountains in the world",
    "FastAPI is used to build Python web backends",
    "Nepal has a population of about 30 million people",
    "PSG won UCL 2 times in a row",
    "Fifa world cup 2018 was won by France"
]

# embed all documents once (this is the "indexing" step)
doc_embeddings = model.encode(documents)

# user asks a question (this is the "query" step)
query = "Who will win the world cup 2026?"
query_embedding = model.encode([query])

# find most similar documents
scores = cosine_similarity(query_embedding, doc_embeddings)[0]
top_indices = np.argsort(scores)[::-1][:2]  # top 2

print(f"Query: {query}\n")
print("Most relevant chunks:")
for i in top_indices:
    print(f"  [{scores[i]:.2f}] {documents[i]}")