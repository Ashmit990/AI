class ChatBox:
    def __init__(self, model):
        self.model = model   
        self.history = []
    
    def send_message(self, message):
        self.history.append({"role": "user", "content": message})
        return self.history
    
    def clear_history(self):
        self.history = []
        return self.history

Bot = ChatBox("Claude-sonnet-2024-06-06")
Bot.send_message("Hello, how are you?")
print(Bot.history)
Bot.clear_history()
print(Bot.history)
print(Bot.model)

