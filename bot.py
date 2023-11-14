print("hello there")
import os
import openai
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
sk-soHSJClAAM54J1BSmb3ET3BlbkFJMnHT2NtHWTekkObntJFJsk-soHSJClAAM54J1BSmb3ET3BlbkFJMnHT2NtHWTekkObntJFJ
#
from llama_index import SimpleDirectoryReader

documents = SimpleDirectoryReader('./').load_data()

from llama_index import LLMPredictor, GPTVectorStoreIndex, PromptHelper
from langchain import OpenAI

# Define LLM
llm_predictor = LLMPredictor(llm=OpenAI(temperature=0.1, model_name="text-davinci-002"))

# Define prompt helper with a reduced number of output tokens
max_input_size = 4096
num_output = 5  # Adjust the number of output tokens to control response length
max_chunk_overlap = 0
prompt_helper = PromptHelper(max_input_size, num_output, max_chunk_overlap)

# Create an index using GPTVectorStoreIndex
custom_LLM_index = GPTVectorStoreIndex(
    documents, llm_predictor=llm_predictor, prompt_helper=prompt_helper
)

query_engine = custom_LLM_index.as_query_engine()
# response = query_engine.query("What do you know about Facebook's LLaMa?")
# print(response)

import openai
import json

class Chatbot:
    def __init__(self, api_key, index):
        self.index = index
        openai.api_key = api_key
        self.chat_history = []

    def generate_response(self, user_input):
        prompt = "\n".join([f"{message['role']}: {message['content']}" for message in self.chat_history[-5:]])
        prompt += f"\nUser: {user_input}"
        response = query_engine.query(user_input)

        message = {"role": "a robot mascot called chacha chaudhary made to provide information to kids in a fun and interactive way using emoji's and funny things", "content": response.response}
        self.chat_history.append({"role": "user", "content": user_input})
        self.chat_history.append(message)
        return message

    def load_chat_history(self, filename):
        try:
            with open(filename, 'r') as f:
                self.chat_history = json.load(f)
        except FileNotFoundError:
            pass

    def save_chat_history(self, filename):
        with open(filename, 'w') as f:
            json.dump(self.chat_history, f)

# Swap out your index below for whatever knowledge base you want
bot = Chatbot("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", index=custom_LLM_index)
bot.load_chat_history("chat_history.json")

while True:
    user_input = input("You: ")
    if user_input.lower() in ["bye", "goodbye"]:
        print("Bot: Goodbye!")
        bot.save_chat_history("chat_history.json")
        break
    response = bot.generate_response(user_input)
    print(f"Bot: {response['content']}")
