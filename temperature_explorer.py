# temperature_explorer.py
# Goal: see exactly how temperature changes LLM output with your own eyes

import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def ask(prompt: str, temperature: float, label: str) -> None:
    """
    Send the same prompt at different temperatures.
    label = just a description so we know what we're testing.
    """
    print(f"\n{'='*55}")
    print(f"Temperature: {temperature} — {label}")
    print(f"{'='*55}")

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant. Keep answers to 2-3 sentences max."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=temperature,  # ← THIS is the only thing changing
        max_tokens=150            # keeping responses short so we can compare easily
    )

    print(response.choices[0].message.content)


# ─────────────────────────────────────────────
# EXPERIMENT 1: Factual question
# A question with one correct answer.
# Watch what happens as temperature rises.
# ─────────────────────────────────────────────
print("\n\n📌 EXPERIMENT 1: Factual question")
print("Prompt: 'What is the capital of Nepal?'")

ask("What is the capital of Nepal?", temperature=0.0, label="Deterministic")
ask("What is the capital of Nepal?", temperature=0.7, label="Balanced")
ask("What is the capital of Nepal?", temperature=1.5, label="High creativity")


# ─────────────────────────────────────────────
# EXPERIMENT 2: Creative question
# A question with no single correct answer.
# Watch temperature HELP here vs hurt above.
# ─────────────────────────────────────────────
print("\n\n📌 EXPERIMENT 2: Creative question")
print("Prompt: 'Give me a unique startup idea for Nepal'")

ask("Give me a unique startup idea for Nepal", temperature=0.0, label="Deterministic")
ask("Give me a unique startup idea for Nepal", temperature=0.7, label="Balanced")
ask("Give me a unique startup idea for Nepal", temperature=1.5, label="High creativity")


# ─────────────────────────────────────────────
# EXPERIMENT 3: Code generation
# Code must be correct. Watch high temperature break it.
# ─────────────────────────────────────────────
print("\n\n📌 EXPERIMENT 3: Code generation")
print("Prompt: 'Write a Python function that returns the sum of a list'")

ask("Write a Python function that returns the sum of a list", temperature=0.0, label="Deterministic")
ask("Write a Python function that returns the sum of a list", temperature=0.7, label="Balanced")
ask("Write a Python function that returns the sum of a list", temperature=1.5, label="High creativity")


# ─────────────────────────────────────────────
# EXPERIMENT 4: Run the SAME prompt 3 times at temp=0
# Should be identical every time — proves determinism
# ─────────────────────────────────────────────
print("\n\n📌 EXPERIMENT 4: Same prompt × 3 at temperature=0")
print("Proving determinism — should be identical")

for i in range(1, 4):
    print(f"\n--- Run {i} ---")
    ask("Name one programming language", temperature=0.0, label=f"Run {i}")


# ─────────────────────────────────────────────
# EXPERIMENT 5: Same prompt 3 times at temp=1.2
# Should be different every time — proves randomness
# ─────────────────────────────────────────────
print("\n\n📌 EXPERIMENT 5: Same prompt × 3 at temperature=1.2")
print("Proving randomness — should vary each time")

for i in range(1, 4):
    print(f"\n--- Run {i} ---")
    ask("Name one programming language", temperature=1.2, label=f"Run {i}")