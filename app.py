import os
import json
import time
import base64
import logging
import tempfile
import hashlib
from pathlib import Path

import requests
import streamlit as st
from dotenv import load_dotenv

# Document parsing
from unstructured.partition.auto import partition

# Text splitting (tetap pakai LangChain splitter, ringan & tidak butuh OpenAI)
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Vector store lokal (FAISS) + embedding sederhana berbasis TF-IDF / BM25
# Kita pakai sentence-transformers agar TIDAK bergantung OpenAI sama sekali
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# ─── KONFIGURASI ─────────────────────────────────────────────────────────────
load_dotenv()

def _get_config(secrets_section: str, key: str, env_key: str, default: str = "") -> str:
    """Baca dari Streamlit Secrets (cloud) atau env var (lokal)."""
    try:
        return st.secrets[secrets_section][key]
    except (KeyError, FileNotFoundError):
        return os.getenv(env_key, default)

GCP_PROJECT_ID = _get_config("app", "GCP_PROJECT_ID", "GCP_PROJECT_ID", "project-raffi-24587")
GCP_LOCATION   = _get_config("app", "GCP_LOCATION",   "GCP_LOCATION",   "us-central1")
GEMINI_MODEL   = _get_config("app", "GEMINI_MODEL",   "GEMINI_MODEL",   "gemini-2.5-flash")
GCP_KEY_PATH   = os.getenv("GCP_KEY_PATH", "gcp-key.json")

TEMP_DIR       = "temp_uploads"
os.makedirs(TEMP_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def get_gcp_access_token() -> str | None:
    from cryptography.hazmat.primitives.serialization import load_pem_private_key
    from cryptography.hazmat.primitives.asymmetric.padding import PKCS1v15
    from cryptography.hazmat.primitives.hashes import SHA256

    # ── Cloud: baca dari Streamlit Secrets ───────────────────────────────
    try:
        private_key  = st.secrets["gcp"]["private_key"]
        client_email = st.secrets["gcp"]["client_email"]
        logger.info("Credentials dari Streamlit Secrets.")
    except (KeyError, FileNotFoundError):
        # ── Lokal: baca dari gcp-key.json ────────────────────────────────
        if not os.path.exists(GCP_KEY_PATH):
            logger.error(f"GCP key tidak ditemukan: {GCP_KEY_PATH}")
            return None
        with open(GCP_KEY_PATH) as f:
            key_data = json.load(f)
        private_key  = key_data.get("private_key")
        client_email = key_data.get("client_email")
        logger.info("Credentials dari file JSON lokal.")

    if not private_key or not client_email:
        logger.error("Struktur GCP credentials tidak valid.")
        return None

    header  = _b64url(json.dumps({"alg": "RS256", "typ": "JWT"}).encode())
    now     = int(time.time())
    payload = _b64url(json.dumps({
        "iss"  : client_email,
        "scope": "https://www.googleapis.com/auth/cloud-platform",
        "aud"  : "https://oauth2.googleapis.com/token",
        "exp"  : now + 3600,
        "iat"  : now,
    }).encode())

    signing_input = f"{header}.{payload}".encode()
    pkey      = load_pem_private_key(private_key.encode(), password=None)
    signature = _b64url(pkey.sign(signing_input, PKCS1v15(), SHA256()))
    jwt_token = f"{header}.{payload}.{signature}"

    resp = requests.post(
        "https://oauth2.googleapis.com/token",
        data={
            "grant_type": "urn:ietf:params:oauth:grant-type:jwt-bearer",
            "assertion" : jwt_token,
        },
        timeout=15,
    )

    if not resp.ok:
        logger.error(f"GCP OAuth gagal: {resp.text}")
        return None

    return resp.json().get("access_token")


# ─────────────────────────────────────────────────────────────────────────────
# GEMINI API CALL
# ─────────────────────────────────────────────────────────────────────────────

def call_gemini(
    access_token: str,
    system_prompt: str,
    user_message: str,
    history: list[dict],
    context: str = "",
    temperature: float = 0.4,
) -> str:
    endpoint = (
        f"https://{GCP_LOCATION}-aiplatform.googleapis.com/v1"
        f"/projects/{GCP_PROJECT_ID}/locations/{GCP_LOCATION}"
        f"/publishers/google/models/{GEMINI_MODEL}:generateContent"
    )

    # Susun system instruction
    sys_instruction = system_prompt
    if context:
        sys_instruction += f"\n\n---\nKONTEKS DOKUMEN (gunakan sebagai referensi utama):\n{context}\n---"

    # Konversi history ke format Gemini
    contents = []
    for msg in history:
        role = "user" if msg["role"] == "user" else "model"
        contents.append({"role": role, "parts": [{"text": msg["content"]}]})

    # Tambahkan pesan user terkini
    contents.append({"role": "user", "parts": [{"text": user_message}]})

    payload = {
        "systemInstruction": {
            "parts": [{"text": sys_instruction}]
        },
        "contents": contents,
        "generationConfig": {
            "temperature"    : temperature,
            "maxOutputTokens": 8192,
            "topP"           : 0.95,
        },
        "safetySettings": [
            {"category": "HARM_CATEGORY_HARASSMENT",        "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH",       "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ],
    }

    resp = requests.post(
        endpoint,
        headers={
            "Authorization": f"Bearer {access_token}",
            "Content-Type" : "application/json",
        },
        json=payload,
        timeout=90,
        verify=True,
    )

    if not resp.ok:
        err = resp.json().get("error", {}).get("message", resp.text)
        logger.error(f"Gemini API [{resp.status_code}]: {err}")
        raise RuntimeError(f"Gemini API Error {resp.status_code}: {err}")

    data = resp.json()
    return data["candidates"][0]["content"]["parts"][0]["text"]


# ─────────────────────────────────────────────────────────────────────────────
# DOCUMENT PROCESSING
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_embedding_model():
    """Load model embedding lokal (berjalan offline, tanpa API key)."""
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def extract_text_from_file(file_path: str) -> str:
    """Ekstrak teks dari PDF, DOCX, PPTX, atau TXT."""
    try:
        elements = partition(filename=file_path)
        return "\n\n".join(e.text for e in elements if e.text)
    except Exception as exc:
        logger.warning(f"partition() gagal untuk {file_path}: {exc}")
        # Fallback: baca sebagai teks biasa
        with open(file_path, "r", errors="ignore") as f:
            return f.read()


def build_vectorstore(file_paths: list[str]) -> FAISS:
    """Bangun vector store dari daftar file yang diunggah."""
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=800,
        chunk_overlap=200,
    )

    all_chunks = []
    for path in file_paths:
        text = extract_text_from_file(path)
        if text.strip():
            chunks = splitter.split_text(text)
            all_chunks.extend(chunks)
            logger.info(f"{Path(path).name}: {len(chunks)} chunk(s)")

    if not all_chunks:
        raise ValueError("Tidak ada teks yang berhasil diekstrak dari dokumen.")

    embeddings = load_embedding_model()
    return FAISS.from_texts(texts=all_chunks, embedding=embeddings)


def retrieve_context(vectorstore: FAISS, question: str, k: int = 5) -> str:
    """Ambil k chunk paling relevan dari vector store."""
    docs = vectorstore.similarity_search(question, k=k)
    return "\n\n".join(d.page_content for d in docs)


def file_list_hash(file_names: list[str]) -> str:
    """Hash nama file untuk deteksi perubahan upload."""
    return hashlib.md5("|".join(sorted(file_names)).encode()).hexdigest()


def save_uploaded_file(uploaded_file) -> str:
    path = os.path.join(TEMP_DIR, uploaded_file.name)
    with open(path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return path


# ─────────────────────────────────────────────────────────────────────────────
# STREAMLIT UI
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """Anda adalah asisten internal perusahaan yang cerdas dan profesional.
Tugas Anda adalah membantu karyawan menjawab pertanyaan berdasarkan dokumen internal yang diberikan.

Panduan:
- Selalu jawab dalam bahasa yang sama dengan pertanyaan pengguna (Indonesia atau Inggris).
- Jika informasi ada di konteks dokumen, gunakan itu sebagai referensi utama.
- Jika tidak ada di dokumen, katakan dengan jujur dan berikan jawaban umum yang membantu.
- Jawab secara ringkas, jelas, dan terstruktur.
- Gunakan format Markdown (bold, bullet, tabel) bila membantu kejelasan.
- Jangan mengarang fakta yang tidak ada di dokumen."""


def init_session():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None
    if "file_hash" not in st.session_state:
        st.session_state.file_hash = None
    if "access_token" not in st.session_state:
        st.session_state.access_token = None
    if "token_expiry" not in st.session_state:
        st.session_state.token_expiry = 0


def get_valid_token() -> str | None:
    now = time.time()
    if st.session_state.access_token and now < st.session_state.token_expiry - 60:
        return st.session_state.access_token

    token = get_gcp_access_token()
    if token:
        st.session_state.access_token = token
        st.session_state.token_expiry = now + 3600
    return token


def render_sidebar() -> list:
    """Render sidebar dan kembalikan daftar path file yang diupload."""
    with st.sidebar:
        st.markdown("## 🤖 Internal Chatbot")

        st.markdown("---")
        st.markdown("## 📂 Upload Dokumen")
        uploaded = st.file_uploader(
            "Pilih satu atau beberapa file",
            type=["pdf", "docx", "pptx", "txt"],
            accept_multiple_files=True,
            key="file_uploader",
        )

        if uploaded:
            st.success(f"✅ {len(uploaded)} file siap diproses")

            # Tombol reset chat
            if st.button("🗑️ Reset Percakapan", use_container_width=True):
                st.session_state.messages = []
                st.rerun()

        st.markdown("---")

    return uploaded or []


def main():
    st.set_page_config(
        page_title="Internal Chatbot",
        page_icon="💬",
        layout="wide",
    )

    st.title("💬 Internal Document Chatbot")
    st.caption("Powered by RASENA Tech")

    init_session()
    uploaded_files = render_sidebar()

    # ── Proses file upload → bangun / perbarui vector store ──────────────────
    if uploaded_files:
        current_hash = file_list_hash([f.name for f in uploaded_files])

        if current_hash != st.session_state.file_hash:
            with st.spinner("⏳ Memproses dan mengindeks dokumen…"):
                try:
                    file_paths = [save_uploaded_file(f) for f in uploaded_files]
                    st.session_state.vectorstore = build_vectorstore(file_paths)
                    st.session_state.file_hash = current_hash
                    st.session_state.messages  = []   # reset chat saat dokumen berubah
                    st.success("✅ Dokumen berhasil diindeks! Silakan mulai bertanya.")
                except Exception as exc:
                    st.error(f"❌ Gagal memproses dokumen: {exc}")
                    logger.exception("build_vectorstore error")

    # ── Tampilkan riwayat chat ────────────────────────────────────────────────
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # ── Input pertanyaan ─────────────────────────────────────────────────────
    placeholder = (
        "Tanyakan sesuatu tentang dokumen yang diupload…"
        if st.session_state.vectorstore
        else "Upload dokumen terlebih dahulu, lalu tanyakan sesuatu…"
    )

    user_input = st.chat_input(placeholder)

    if user_input:
        # Tampilkan pesan user
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Generate respons
        with st.chat_message("assistant"):
            with st.spinner("🤔 Sedang berpikir…"):
                try:
                    # 1. Dapatkan access token (cached / baru)
                    token = get_valid_token()
                    if not token:
                        st.error("❌ Gagal mendapatkan GCP access token. Periksa file gcp-key.json.")
                        st.stop()

                    # 2. Ambil konteks relevan dari vector store (jika ada)
                    context = ""
                    if st.session_state.vectorstore:
                        context = retrieve_context(
                            st.session_state.vectorstore, user_input, k=5
                        )

                    # 3. Kirim ke Gemini (history tanpa pesan terkini yang baru ditambahkan)
                    history_for_api = st.session_state.messages[:-1]  # exclude pesan user terkini
                    reply = call_gemini(
                        access_token = token,
                        system_prompt= SYSTEM_PROMPT,
                        user_message = user_input,
                        history      = history_for_api,
                        context      = context,
                    )

                    st.markdown(reply)
                    st.session_state.messages.append({"role": "assistant", "content": reply})

                except RuntimeError as exc:
                    err_msg = str(exc)
                    st.error(f"❌ {err_msg}")
                    logger.error(err_msg)
                except Exception as exc:
                    st.error(f"❌ Terjadi kesalahan tak terduga: {exc}")
                    logger.exception("Unexpected error during chat")


if __name__ == "__main__":
    main()
