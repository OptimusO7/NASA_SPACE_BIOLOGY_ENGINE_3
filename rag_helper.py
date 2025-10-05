"""
🧠 RAG Helper – Gemini v2 + Memory (NASA Edition)
------------------------------------------------
Retrieves relevant research chunks from ChromaDB and asks Gemini v2 (google-genai)
with short-term memory across the current chat session.
"""

import os
import chromadb
from chromadb.utils import embedding_functions
from google import genai

# === CONFIG ===
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "nasa_papers"
EMBED_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
GEMINI_MODEL = "models/gemini-2.0-flash"

SYSTEM_PROMPT = (
    "You are the **NASA Space Biology Assistant**, an intelligent research AI created by "
    "**Osborn Nartey** from **Team DarkSun** during the **2025 NASA Space Apps Challenge**. "
    "Your mission is to assist scientists, engineers, and students in exploring insights "
    "from NASA’s space biology research data.\n\n"
    "Guidelines:\n"
    "- Be professional, concise, and scientifically accurate.\n"
    "- When possible, explain biological effects of microgravity and spaceflight.\n"
    "- If information is unclear or missing, say: 'I don’t know based on the available research.'\n"
    "- Refer to NASA studies naturally when appropriate (e.g., 'According to NASA research...')."
)


class RAGRetriever:
    def __init__(self):
        # Load API key
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("❌ No Gemini API key found. Set GEMINI_API_KEY or GOOGLE_API_KEY.")
        print("🔐 Using Gemini API key loaded successfully.")

        # Initialize Gemini client
        self.client = genai.Client(api_key=api_key)

        # Connect to Chroma vector store
        self.chroma = chromadb.PersistentClient(path=CHROMA_DIR)
        self.embed_func = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL_NAME)
        self.collection = self.chroma.get_collection(name=COLLECTION_NAME, embedding_function=self.embed_func)
        print(f"📚 Loaded Chroma collection: {COLLECTION_NAME}")

        # Initialize in-memory conversation buffer
        self.memory = []

    def retrieve_context(self, query, top_k=4):
        """Retrieve top-k relevant context documents"""
        results = self.collection.query(query_texts=[query], n_results=top_k)
        docs, sources = [], []
        for i in range(len(results["documents"][0])):
            docs.append(results["documents"][0][i])
            sources.append(results["metadatas"][0][i].get("source", "Unknown source"))
        return docs, sources

    def _build_prompt(self, query, context):
        """Builds full prompt with memory + system + context"""
        memory_text = "\n".join([f"User: {q}\nAssistant: {a}" for q, a in self.memory[-5:]])  # last 5 exchanges
        return (
            f"{SYSTEM_PROMPT}\n\n"
            f"Conversation so far:\n{memory_text}\n\n"
            f"Context from NASA research:\n{context}\n\n"
            f"User Question: {query}\n"
            f"Assistant:"
        )

    def ask_gemini(self, query):
        """Full RAG + Memory pipeline"""
        docs, sources = self.retrieve_context(query)
        context = "\n\n---\n\n".join(docs)
        prompt = self._build_prompt(query, context)

        try:
            response = self.client.models.generate_content(
                model=GEMINI_MODEL,
                contents=[prompt]
            )
            answer = getattr(response, "text", str(response)).strip()
        except Exception as e:
            answer = f"⚠️ Gemini API error: {e}"

        # Update memory (retain last 10 turns)
        self.memory.append((query, answer))
        if len(self.memory) > 10:
            self.memory = self.memory[-10:]

        return answer, sources
