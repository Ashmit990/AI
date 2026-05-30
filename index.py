import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# Note: Switched from Gemini to Groq as requested
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

model_id = "llama-3.3-70b-versatile"

class ChatBot:

    def __init__(self,model):
     self.client = client
     self.model_id = model
     self.history=[]

    def chat(self,message):
     self.history.append({"role": "user", "content": message})
     try:
       response = self.client.chat.completions.create(
         model=self.model_id,
         messages = self.history
       )

       reply = response.choices[0].message.content
       self.history.append({"role": "assistant", "content": reply})

       return reply
     except Exception as e:
       return f"Error: {str(e)}"

bot = ChatBot(model_id)
print(bot.chat("Hello, How r u groq?"))
print(bot.chat("What can you do?"))
print(bot.chat("Can you tell me a joke?"))
print(bot.chat("What did I say first?"))

