"""
📘 NASA Research Paper Index Builder (Chroma + SentenceTransformer)
-------------------------------------------------------------------
This script scans ./papers, embeds their text using all-mpnet-base-v2,
and stores everything in a persistent ChromaDB collection.
"""

import os
from PyPDF2 import PdfReader
from tqdm import tqdm
import chromadb
from chromadb.utils import embedding_functions

# === CONFIG ===
PDF_DIR = "papers"
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "nasa_papers"
EMBED_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"

# === SETUP ===
if not os.path.exists(PDF_DIR):
    raise FileNotFoundError(f"❌ No folder named '{PDF_DIR}' found. Place your PDFs there first.")

client = chromadb.PersistentClient(path=CHROMA_DIR)
embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL_NAME)

# Delete old collection (clean rebuild)
try:
    client.delete_collection(COLLECTION_NAME)
    print(f"🧹 Old collection '{COLLECTION_NAME}' deleted.")
except Exception:
    pass

collection = client.create_collection(name=COLLECTION_NAME, embedding_function=embedding_func)
print(f"✨ Created new collection: {COLLECTION_NAME}")
print(f"🔍 Using embedding model: {EMBED_MODEL_NAME}")

# === FUNCTION: Extract text ===
def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, "rb") as f:
        reader = PdfReader(f)
        for page in reader.pages:
            text += page.extract_text() or ""
    return text.strip()

# === INDEX ALL PDFs ===
pdf_files = [f for f in os.listdir(PDF_DIR) if f.endswith(".pdf")]
print(f"[✓] Loaded {len(pdf_files)} PDFs total.")

docs, metadatas, ids = [], [], []
chunk_size, chunk_overlap = 1000, 100

def chunk_text(text, size=chunk_size, overlap=chunk_overlap):
    return [text[i:i+size] for i in range(0, len(text), size - overlap)]

id_counter = 0
for pdf_file in tqdm(pdf_files, desc="📄 Processing PDFs"):
    pdf_path = os.path.join(PDF_DIR, pdf_file)
    text = extract_text_from_pdf(pdf_path)
    chunks = chunk_text(text)
    for i, chunk in enumerate(chunks):
        docs.append(chunk)
        metadatas.append({"source": pdf_file, "chunk": i})
        ids.append(str(id_counter))
        id_counter += 1

print(f"\n🧩 New chunks to add: {len(docs)}")
collection.add(documents=docs, metadatas=metadatas, ids=ids)

print("\n✅ Successfully updated Chroma index.")
print(f"📈 Total stored chunks: {len(ids)}")
print(f"📂 Database location: {os.path.abspath(CHROMA_DIR)}")
print(f"📚 Collection name: {COLLECTION_NAME}")
print("✨ Ready to query with app_2.py!")
