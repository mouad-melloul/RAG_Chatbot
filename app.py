from flask import Flask, render_template, request, jsonify, redirect, url_for
import os
from werkzeug.utils import secure_filename
from langchain_community.vectorstores import FAISS
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.document_loaders import PyMuPDFLoader
import faiss
from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnablePassthrough
import tiktoken
from dotenv import load_dotenv
import shutil

# Configuration initiale
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
load_dotenv()

# Configuration pour le téléversement de fichiers
UPLOAD_FOLDER = 'temp_uploads'
ALLOWED_EXTENSIONS = {'pdf'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Vérifie si l'extension du fichier est autorisée
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Nettoyer les fichiers temporaires
def cleanup_temp_files():
    if os.path.exists(UPLOAD_FOLDER):
        shutil.rmtree(UPLOAD_FOLDER)
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialiser le système RAG avec un fichier PDF
def initialize_rag_system(file_path):
    # Charger le fichier PDF
    loader = PyMuPDFLoader(file_path)
    pages = loader.load()
    
    # Découper le document en chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(pages)
    
    # Filtrer les chunks vides
    chunks = [chunk for chunk in chunks if chunk.page_content.strip()]
    
    if not chunks:
        raise ValueError("No valid chunks were generated from the PDF.")
    
    # Générer les embeddings
    embeddings = OllamaEmbeddings(model='nomic-embed-text', base_url="http://localhost:11434")
    single_vector = embeddings.embed_query("this is some text data")
    print(f"Shape of single vector: {len(single_vector)}")
    
    index = faiss.IndexFlatL2(len(single_vector))
    
    # Créer le vector store
    vector_store = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={}
    )
    
    # Ajouter les documents au vector store
    ids = vector_store.add_documents(documents=chunks)
    print(f"Documents added to vector store. IDs: {ids}")
    
    # Créer le retriever
    retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={'k': 3, 'fetch_k': 100, 'lambda_mult': 1})
    
    # Charger le modèle de langage
    model = ChatOllama(model="llama3.2:1b", base_url="http://localhost:11434")
    
    # Définir le prompt
    prompt = """
        You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question.
        If you don't know the answer, just say that you don't know.
        Answer in bullet points. Make sure your answer is relevant to the question and it is answered from the context only.
        Question: {question}
        Context: {context}
        Answer:
    """
    prompt = ChatPromptTemplate.from_template(prompt)
    
    # Créer la chaîne RAG
    def format_docs(docs):
        return "\n\n".join([doc.page_content for doc in docs])
    
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )
    
    return rag_chain

# Variable globale pour stocker la chaîne RAG
rag_chain = None

# Route pour la page d'accueil
@app.route("/")
def index():
    return render_template("chat.html")

# Route pour téléverser un fichier PDF
@app.route("/upload", methods=["GET", "POST"])
def upload_file():
    global rag_chain
    
    if request.method == "POST":
        if 'file' not in request.files:
            return jsonify({"error": "No file part"}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            # Initialiser le système RAG avec le fichier PDF téléversé
            try:
                rag_chain = initialize_rag_system(file_path)
                return jsonify({"message": "PDF uploaded and processed successfully."}), 200
            except Exception as e:
                return jsonify({"error": f"Failed to process PDF: {str(e)}"}), 500
    
    return render_template("upload.html")

# Route pour poser une question
@app.route("/ask", methods=["POST"])
def ask_question():
    global rag_chain
    
    try:
        data = request.get_json()
        question = data.get("question", "")
        if not question:
            return jsonify({"error": "No question provided."}), 400
        
        if rag_chain is None:
            return jsonify({"error": "Please upload a PDF file first."}), 400
        
        # Traiter la question via la chaîne RAG
        response = rag_chain.invoke(question)
        return jsonify({"answer": response})
    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

# Démarrer l'application
if __name__ == "__main__":
    cleanup_temp_files()
    app.run(debug=True)