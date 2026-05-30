import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

class ChatBot:

    def __init__(self, model, system_prompt=None,
                 strategy="sliding", max_messages=6):
        self.client       = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model        = model
        self.system_prompt = system_prompt
        self.strategy     = strategy    # "truncation" | "sliding" | "summarization"
        self.max_messages = max_messages
        self.history      = []          # full conversation always stored here
        self.summary      = None        # only used by summarization strategy

    # ── STRATEGY 1: Truncation ──────────────────────────────────
    def apply_truncation(self):
        if len(self.history) > self.max_messages:
            self.history = self.history[-self.max_messages:]  # deletes old msgs

    # ── STRATEGY 2: Sliding Window ──────────────────────────────
    def get_windowed_messages(self):
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages += self.history[-self.max_messages:]         # slice, don't delete
        return messages

    # ── STRATEGY 3: Summarization ───────────────────────────────
    def summarize_old_messages(self, old_messages):
        prompt = f"""Summarize this conversation in 3 sentences.
Keep: user's name, goals, key facts.
{old_messages}"""
        res = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}]
        )
        return res.choices[0].message.content

    def get_summarized_messages(self):
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        if self.summary:
            messages.append({"role": "system",
                             "content": f"Previous summary: {self.summary}"})
        messages += self.history[-self.max_messages:]
        return messages

    # ── MAIN CHAT METHOD ────────────────────────────────────────
    def chat(self, user_message):
        self.history.append({"role": "user", "content": user_message})

        # pick which strategy to use
        if self.strategy == "truncation":
            self.apply_truncation()
            messages_to_send = self.history

        elif self.strategy == "sliding":
            messages_to_send = self.get_windowed_messages()

        elif self.strategy == "summarization":
            if len(self.history) > self.max_messages:
                old = self.history[:-self.max_messages]
                self.summary = self.summarize_old_messages(str(old))
            messages_to_send = self.get_summarized_messages()

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages_to_send
            )
            reply = response.choices[0].message.content
            self.history.append({"role": "assistant", "content": reply})
            return reply
        except Exception as e:
            return f"Error: {str(e)}"


# ── TEST ALL 3 STRATEGIES ───────────────────────────────────
msgs = [
    "Hi, my name is Arun",
    "I am learning AI engineering",
    "I know JavaScript",
    "My goal is a job in 6 months",
    "What is RAG?",
    "What is my name?",    # will it remember?
]

for strategy in ["truncation", "sliding", "summarization"]:
    print(f"\n=== {strategy.upper()} ===")
    bot = ChatBot("llama-3.3-70b-versatile",
                  system_prompt="You are a helpful assistant.",
                  strategy=strategy, max_messages=4)
    for msg in msgs:
        reply = bot.chat(msg)
        print(f"User: {msg}")
        print(f"Bot:  {reply[:80]}...\n")