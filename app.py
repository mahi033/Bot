# 7/22/2024
# Copied from mis space and modified for msba space

# 5/1/2024

# This version added saving chat history to a log file (need persist data from a space to a dataset)
# Updated the GPT model to gpt-4
# Add timestamp and ip address
# upgrade llama-index to version 0.10: migrate from ServiceContext to Settings

# 2/23/2024
# This version uses different method in llama index to define llm model
# Removed deprecated classes and replaced with newest dependencies

# Start by setting token and debug mode before starting schedulers
import os
from huggingface_hub import logging, login

# The access token must be saved in the secrets of this space first
#login(token=os.environ.get("new_data_token"), write_permission=True)
#login(token=os.environ.get("data_token")) # this is a new fine_grained token

#login(token=os.getenv("data_token")) # this is a new fine_grained token

login(token=os.getenv("data_token_msba"), write_permission=True)
#logging.set_verbosity_debug()

import openai
import json
import gradio as gr
from openai import OpenAI

# rebuild storage context and load knowledge index
from llama_index import StorageContext, load_index_from_storage, LLMPredictor, ServiceContext
from llama_index.llms import OpenAI

# for llama-index 0.10
#from llama_index.core import StorageContext
#from llama_index.core import load_index_from_storage
#from llama_index.llms.openai import OpenAI
#from llama_index.core import Settings

# add datetime and ip to the log file
from datetime import datetime;
import socket;

# access data folder of persistent storage
from pathlib import Path
from huggingface_hub import CommitScheduler
from uuid import uuid4

# generate an unique identifier for the session
session_id = uuid4()

# deprecated (llama-index 0.9)
storage_context = StorageContext.from_defaults(persist_dir='./')
# gpt-3.5-turbo is the current default model
llm = OpenAI(temperature=0.5, model_name="gpt-4")
service_context = ServiceContext.from_defaults(llm=llm)
index = load_index_from_storage(storage_context, service_context=service_context)
# for llama-index 0.10
#Settings.llm = OpenAI(temperature=0.5, model="gpt-3.5_turbo")
#index = load_index_from_storage(storage_context)

class Chatbot:
    def __init__(self, api_key, index):
        self.index = index
        openai.api_key = api_key
        self.chat_history = []

        # set chat history data path in data folder (persistent storage)
        dataset_dir = Path("logs")
        dataset_dir.mkdir(parents=True, exist_ok=True)
        #self.dataset_path = dataset_dir / f"chat_log_{uuid4()}.json"
        self.dataset_path = dataset_dir / f"chat_log_{session_id}.json"
        

        self.scheduler = CommitScheduler(
            repo_id="history_data",
            repo_type="dataset",
            folder_path=dataset_dir,
            #path_in_repo="data",
            path_in_repo="data_msba",
        )
        
    def generate_response(self, user_input):
        query_engine = index.as_query_engine()
        response = query_engine.query(user_input)
        
        # generate response
        message = {"role": "assistant", "content": response.response}
        
        return message
    
    # do not need this function if use append mode when dump data in file
    #def load_chat_history(self):
    #    try:
    #        with open(self.dataset_path, 'r') as f:
    #            self.chat_history = json.load(f)
    #    except FileNotFoundError:
    #        pass
    
    def append_chat_history(self, user_input, output):
            # create a dictionary for the chat history
        #self.chat_history = []
        dt = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        #print(dt)
        
        #hostname = socket.gethostname()  # this returns server hostname
        #ip = socket.gethostbyname(hostname)
        client_socket = socket.socket()
        client_socket.connect(("huggingface.co",80))
        ip = client_socket.getpeername()[0]
        #print(ip)
        
        #self.chat_history.append({"role": "datetime", "content": dt})
        #self.chat_history.append({"role": "IP", "content": ip})
        #self.chat_history.append({"role": "user", "content": user_input})
        #self.chat_history.append({"role": "assistant", "content": output})
        
        # save the data in dictionary format
        dictionary = {
            "datetime": dt,
            "ip": ip,
            "user": user_input,
            "assistant": output
        }
        self.chat_history.append(dictionary)

    def save_chat_history(self):
        with self.scheduler.lock:
            with self.dataset_path.open("a") as f:
                json.dump(self.chat_history, f)
                f.write("\n")

def create_bot(user_input):
    bot = Chatbot(os.getenv("OPENAI_API_KEY"), index=index)
    #bot.load_chat_history();
    
    if user_input:
         # use moderations endpoint to check input
        client = openai.OpenAI()
        response_mod = client.moderations.create(input=user_input)
        response_dict = response_mod.model_dump()
        flagged = response_dict['results'][0]['flagged']
        #print("Flagged:", flagged)
    
        if not flagged:
            response_bot = bot.generate_response(user_input)
            output = response_bot['content']
        else:
             output = "Invalid request."
        
        bot.append_chat_history(user_input, output)
        bot.save_chat_history()
        
        return output

inputs = gr.components.Textbox(lines=7, label="You can ask any questions related to the general information or content of the course. For example, what is the assignment late policy, what is ETL process, etc.")
outputs = gr.components.Textbox(label="Response")

gr.Interface(fn=create_bot, inputs=inputs, outputs=outputs, title="Educational Chatbot",
             description="This is a virtual learning assistant designed for MBA 751 (Beta version 2.0, powered by GPT-4).\nNote: Chatbot can make mistakes. Consider checking important information."
            ).launch(share=True)
