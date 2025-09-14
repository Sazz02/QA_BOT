import os
import hashlib
import pickle
import tempfile
import shutil
import traceback

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename

# LangChain + Groq imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq

# -------------------- Config --------------------
CACHE_DIR = "vector_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# Default Groq model (set to the one you confirmed works)
DEFAULT_GROQ_MODEL = os.environ.get("GROQ_MODEL_NAME", "llama2-70b-4096")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")  # must be set in Render secrets
ALLOWED_EXTENSIONS = {"pdf"}

# -------------------- Flask App --------------------
app = Flask(__name__, static_folder="static", static_url_path="/static")
CORS(app)  # allow browser requests

# -------------------- Utilities --------------------
def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def get_pdf_hash(pdf_path: str) -> str:
    with open(pdf_path, "rb") as f:
        data = f.read()
    return hashlib.md5(data).hexdigest()

def build_vectorstore(pdf_path: str):
    """Load PDF, chunk, embed and build FAISS vectorstore."""
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore

def get_vectorstore(pdf_path: str):
    pdf_hash = get_pdf_hash(pdf_path)
    cache_file = os.path.join(CACHE_DIR, f"{pdf_hash}.pkl")
    if os.path.exists(cache_file):
        try:
            with open(cache_file, "rb") as f:
                return pickle.load(f)
        except Exception:
            os.remove(cache_file)
    vs = build_vectorstore(pdf_path)
    with open(cache_file, "wb") as f:
        pickle.dump(vs, f)
    return vs

def get_answer_from_pdf(pdf_path: str, question: str) -> str:
    """Return an answer string. Uses Groq if key is set, else returns mock answer."""
    vs = get_vectorstore(pdf_path)
    retriever = vs.as_retriever(search_kwargs={"k": 3})

    if GROQ_API_KEY:
        llm = ChatGroq(
            groq_api_key=GROQ_API_KEY,
            model_name=DEFAULT_GROQ_MODEL,
        )
        qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
        return qa.run(question)
    else:
        docs = retriever.get_relevant_documents(question)
        combined = "\n\n".join([d.page_content for d in docs[:3]])
        return "MOCK ANSWER (no GROQ_API_KEY set). Retrieved content:\n\n" + combined

# -------------------- Routes --------------------
@app.route("/", methods=["GET"])
def home():
    return send_from_directory("static", "index.html")

@app.route("/api/ask", methods=["POST"])
def api_ask():
    """
    Expects multipart/form-data:
      - file: pdf file
      - question: text
    Returns JSON: { "answer": "...", "error": null }
    """
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file provided"}), 400

        file = request.files["file"]
        question = request.form.get("question", "").strip()

        if file.filename == "":
            return jsonify({"error": "Empty filename"}), 400
        if not allowed_file(file.filename):
            return jsonify({"error": "Only PDF files are allowed"}), 400
        if not question:
            return jsonify({"error": "Question is required"}), 400

        filename = secure_filename(file.filename)
        tmpdir = tempfile.mkdtemp()
        filepath = os.path.join(tmpdir, filename)
        file.save(filepath)

        answer = get_answer_from_pdf(filepath, question)

        try:
            if os.path.exists(filepath):
                os.remove(filepath)
            shutil.rmtree(tmpdir, ignore_errors=True)
        except Exception:
            pass

        return jsonify({"answer": answer, "error": None})
    except Exception as e:
        tb = traceback.format_exc()
        return jsonify({"answer": None, "error": str(e), "trace": tb}), 500

@app.route("/healthz", methods=["GET"])
def healthz():
    return jsonify({"status": "ok"})

# -------------------- Run --------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
