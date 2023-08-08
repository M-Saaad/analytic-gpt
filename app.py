from flask import Flask, render_template, request
from langchain.document_loaders import CSVLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.agents import create_csv_agent
from langchain.agents.agent_types import AgentType
import os
from dotenv import load_dotenv
load_dotenv()

app = Flask(__name__, static_folder='static', template_folder='templates')

# loader = CSVLoader(file_path='data/test_data.csv')
# index_creator = VectorstoreIndexCreator()
# docsearch = index_creator.from_loaders([loader])
# chain = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=docsearch.vectorstore.as_retriever(), input_key="question")
agent = create_csv_agent(OpenAI(temperature=0), 'data/test_data.csv', verbose=True,agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION)

def generate_response(query):
    response = agent.run(query)
    return response

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat_ai", methods=["POST"])
def chat():
    user_message = request.form["user_input"]
    chat_history = request.form["chat_history"]
    prompt = f"{chat_history}User: {user_message}\nAI: "
    response = generate_response(prompt)
    chat_history = f"{chat_history}User: {user_message}\nAI: {response}\n"
    return render_template("index.html", chat_history=chat_history)

if __name__ == "__main__":
    app.run(debug=True)


"""
Based on the provided code snippet and context, here is one way to modify the RetrievalQA chain to remember old context when answering new questions:

from langchain import ConversationChain, OpenAI
from langchain.memories import ConversationBufferMemory

# Create memory
memory = ConversationBufferMemory(memory_key="conversation") 

# Create RetrievalQA chain
chain = RetrievalQA.from_chain_type(
    llm=OpenAI(),
    chain_type="stuff",
    retriever=docsearch.vectorstore.as_retriever(),
    input_key="question"  
)

# Wrap in ConversationChain to track context
conv_chain = ConversationChain(chain=chain, memory=memory)

# Sample usage
response1 = conv_chain.predict(question="What is the capital of France?") 

response2 = conv_chain.predict(question="What is its population?")
The key steps are:

Create a ConversationBufferMemory to store conversation history

Create the normal RetrievalQA chain

Wrap the RetrievalQA chain in a ConversationChain and pass the memory

Each call to predict will now save context to memory

The chain can leverage the conversation context when answering new questions

This allows the RetrievalQA chain to maintain state and remember previous questions/answers to have more natural conversations.
"""