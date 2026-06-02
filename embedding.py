from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import sys

try:
    model=SentenceTransformer("all-MiniLM-L6-v2")
except Exception as e:
    print(e)
    sys.exit()

sentences = [
    "I love cats",
    "I love dogs",
    "Python is a programming language",
    "Kathmandu is in Nepal"
]

embedding=model.encode(sentences)
scores=cosine_similarity([embedding[0]],embedding[0:])
print(scores)

print(f"similarity of I love cats: {scores[0][0]:.2f}")
print(f"similarity of I love dogs: {scores[0][1]:.2f}")
print(f"similarity of Python is a programming language: {scores[0][2]:.2f}")
print(f"similarity of Kathmandu is in Nepal: {scores[0][3]:.2f}")
