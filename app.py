from flask import Flask, render_template, request
from langchain.document_loaders import CSVLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
import os
# Put your open ai key in evironment variable OPENAI_API_KEY
from dotenv import load_dotenv
load_dotenv()

app = Flask(__name__, static_folder='static', template_folder='templates')

loader = CSVLoader(file_path='data/test_data.csv')
index_creator = VectorstoreIndexCreator()
docsearch = index_creator.from_loaders([loader])
chain = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=docsearch.vectorstore.as_retriever(), input_key="question")

def generate_response(query):
    response = chain({"question": query})
    return response['result']

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