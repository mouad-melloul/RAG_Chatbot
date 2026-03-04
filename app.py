from flask import Flask, render_template, request, jsonify
import os
from werkzeug.utils import secure_filename
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.document_loaders import PyMuPDFLoader
import faiss
from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv
import shutil
import warnings

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
warnings.filterwarnings("ignore")

load_dotenv()

UPLOAD_FOLDER = 'temp_uploads'
ALLOWED_EXTENSIONS = {'pdf'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# ---------------------------------------------------------------
# Global state
# ---------------------------------------------------------------
rag_chain    = None
vector_store = None
embeddings   = None


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def cleanup_temp_files():
    if os.path.exists(UPLOAD_FOLDER):
        shutil.rmtree(UPLOAD_FOLDER)
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def initialize_rag_system(file_path):
    global vector_store, embeddings

    # Load PDF
    loader = PyMuPDFLoader(file_path)
    pages = loader.load()

    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(pages)

    # Filter empty chunks
    chunks = [chunk for chunk in chunks if chunk.page_content.strip()]

    if not chunks:
        raise ValueError("No valid chunks were generated from the PDF.")

    # Reuse embeddings instance if already created
    if embeddings is None:
        embeddings = OllamaEmbeddings(model='nomic-embed-text', base_url="http://localhost:11434")

    single_vector = embeddings.embed_query("this is some text data")
    print(f"Shape of single vector: {len(single_vector)}")

    # Build a FAISS index for this PDF
    index = faiss.IndexFlatL2(len(single_vector))
    new_store = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={}
    )

    ids = new_store.add_documents(documents=chunks)
    print(f"Documents added to vector store. IDs: {ids}")

    # Merge into global store for multi-PDF support
    if vector_store is None:
        vector_store = new_store
        print("Global vector store created.")
    else:
        vector_store.merge_from(new_store)
        print("New PDF merged into existing vector store.")

    # Retriever on the global store (contains ALL uploaded PDFs)
    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={
            'k': 5,
            'fetch_k': 20,
            'lambda_mult': 0.5
        }
    )

    # Language model
    model = ChatOllama(model="llama3.2:1b", base_url="http://localhost:11434")

    # Simple prompt — no history

    prompt_template = """
        You are a helpful and accurate document assistant.
        Your job is to answer questions based strictly on the context provided below.

        Guidelines:
        - Answer clearly and directly using only what is written in the context
        - If the answer is a list, present it in a clean numbered or bulleted format
        - Do not add commentary, assumptions, or analysis beyond what was asked
        - Do not use filler phrases like "it appears" or "based on the context"
        - If the information is not in the context, simply say "This information is not mentioned in the document"
        - Keep answers focused — do not add unrequested sections or conclusions

        Context: {context}

        Question: {question}

        Answer:
    """
    prompt = ChatPromptTemplate.from_template(prompt_template)

    def format_docs(docs):
        return "\n\n".join([doc.page_content for doc in docs])

    rag_chain_local = (
        {
            "context":  retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | model
        | StrOutputParser()
    )

    return rag_chain_local


# ---------------------------------------------------------------
# Routes
# ---------------------------------------------------------------
@app.route("/")
def index():
    return render_template("chat.html")


@app.route("/upload", methods=['POST'])
def upload_file():
    global rag_chain

    try:
        if 'pdf_file' not in request.files:
            return jsonify({"error": "No file part"}), 400

        files = request.files.getlist('pdf_file')

        if not files or all(f.filename == '' for f in files):
            return jsonify({"error": "No selected file"}), 400

        results = []
        for file in files:
            if file.filename == '':
                continue
            if not allowed_file(file.filename):
                results.append({"file": file.filename, "error": "Not a PDF, skipped"})
                continue

            filename  = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            print(f"File saved to: {file_path}")

            rag_chain = initialize_rag_system(file_path)
            results.append({"file": filename, "status": "ok"})

        return jsonify({
            'message': f'{len(results)} file(s) uploaded successfully',
            'files':   results
        }), 200

    except Exception as e:
        print(f"Upload error: {str(e)}")
        return jsonify({'error': 'Server error during upload'}), 500


@app.route("/ask", methods=["POST"])
def ask_question():
    global rag_chain

    try:
        data     = request.get_json()
        question = data.get("question", "")

        if not question:
            return jsonify({"error": "No question provided."}), 400
        if rag_chain is None:
            return jsonify({"error": "Please upload a PDF file first."}), 400

        response = rag_chain.invoke(question)

        return jsonify({"answer": response})

    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500


@app.route("/clear", methods=["POST"])
def clear_conversation():
    return jsonify({"message": "Conversation cleared."}), 200


# ---------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------
if __name__ == "__main__":
    cleanup_temp_files()
    app.run(debug=True)