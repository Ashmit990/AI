import os
import json
from groq import Groq
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

class ChatBot:

    def __init__(self, model, system_prompt=None,
                 strategy="sliding", max_messages=6):
        self.client        = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model         = model
        self.system_prompt = system_prompt
        self.strategy      = strategy
        self.max_messages  = max_messages
        self.history       = []
        self.summary       = None

    def apply_truncation(self):
        if len(self.history) > self.max_messages:
            self.history = self.history[-self.max_messages:]

    def get_windowed_messages(self):
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages += self.history[-self.max_messages:]
        return messages

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

    def chat(self, user_message):
        self.history.append({"role": "user", "content": user_message})

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


# ── HELPER FUNCTIONS ─────────────────────────────────────────

def print_help():
    print("\n--- Available Commands ---")
    print("  /history  → print conversation so far")
    print("  /clear    → reset conversation")
    print("  /save     → save conversation to JSON file")
    print("  /help     → show this menu")
    print("  /quit     → exit the chatbot")
    print("--------------------------\n")

def save_conversation(bot):
    if len(bot.history) == 0:
        print("Nothing to save yet.\n")
        return

    filename = f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    data = {
        "strategy":       bot.strategy,
        "model":          bot.model,
        "total_messages": len(bot.history),
        "saved_at":       datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "conversation":   bot.history
    }
    with open(filename, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Conversation saved to {filename}\n")

def print_history(bot):
    if len(bot.history) == 0:
        print("No history yet.\n")
        return
    print("\n--- Conversation History ---")
    for msg in bot.history:
        role    = msg["role"].upper()
        content = msg["content"]
        print(f"{role}: {content}\n")
    print("----------------------------\n")


# ── MAIN PROGRAM ─────────────────────────────────────────────

print("=" * 40)
print("   Welcome to Smart Chatbot")
print("=" * 40)
print("\nChoose a strategy:")
print("  truncation     → deletes old messages")
print("  sliding        → keeps recent window")
print("  summarization  → compresses old context")

strategy = input("\nEnter strategy: ").strip().lower()

if strategy not in ["truncation", "sliding", "summarization"]:
    print("Invalid strategy, defaulting to sliding")
    strategy = "sliding"

print(f"\nUsing strategy: {strategy}")
print("Type /help to see all commands\n")

bot = ChatBot(
    model="llama-3.3-70b-versatile",
    system_prompt="You are a helpful and friendly assistant.",
    strategy=strategy,
    max_messages=6
)

while True:
    user_input = input("You: ").strip()

    if not user_input:          # user pressed Enter with nothing typed
        continue

    elif user_input == "/quit":
        save_conversation(bot)  # auto-save on exit
        print("Goodbye!")
        break

    elif user_input == "/clear":
        bot.history = []
        bot.summary = None
        print("Conversation cleared!\n")

    elif user_input == "/history":
        print_history(bot)

    elif user_input == "/save":
        save_conversation(bot)

    elif user_input == "/help":
        print_help()

    else:
        reply = bot.chat(user_input)
        print(f"Bot: {reply}\n")