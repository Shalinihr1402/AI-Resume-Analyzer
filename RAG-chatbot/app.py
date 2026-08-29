import os
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from ingest import process_pdf
from langchain.vectorstores import FAISS
from langchain.embeddings import OllamaEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import Ollama

load_dotenv()

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = int(os.getenv('MAX_CONTENT_LENGTH', 10 * 1024 * 1024))

# Global vector store (in-memory for simplicity)
vector_store = None

def get_qa_chain():
    global vector_store
    embeddings = OllamaEmbeddings(model=os.getenv('OLLAMA_MODEL', 'llama2'))
    # Reuse existing FAISS index if already built
    if vector_store is None:
        return None
    llm = Ollama(model=os.getenv('OLLAMA_MODEL', 'llama2'))
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=vector_store.as_retriever())
    return qa

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files.get('pdf')
    if not file:
        return jsonify({'error': 'No file uploaded'}), 400
    # Save temporarily
    upload_path = os.path.join('uploads', file.filename)
    os.makedirs('uploads', exist_ok=True)
    file.save(upload_path)
    # Process PDF and build vector store
    global vector_store
    vector_store = process_pdf(upload_path)
    return jsonify({'message': 'PDF processed successfully'})

@app.route('/ask', methods=['POST'])
def ask():
    data = request.get_json()
    question = data.get('question')
    if not question:
        return jsonify({'error': 'No question provided'}), 400
    qa = get_qa_chain()
    if qa is None:
        return jsonify({'error': 'No documents indexed'}), 400
    answer = qa.run(question)
    return jsonify({'answer': answer})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
