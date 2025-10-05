"""
🚀 NASA BioDash Backend – RAG + Analytics + Graphs
-------------------------------------------------
Provides APIs for:
- Overview stats
- Category / Mission / Timeline analytics
- AI-powered insights (Gemini)
- PNG chart endpoints for frontend visualizations
"""

import os
import re
import io
import pandas as pd
import matplotlib.pyplot as plt
import chromadb
from chromadb.utils import embedding_functions
from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from rag_helper import RAGRetriever

# === CONFIG ===
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "nasa_papers"
EMBED_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"

# === FASTAPI APP ===
app = FastAPI(title="NASA BioDash API", version="2.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === INIT RAG + DB ===
rag = RAGRetriever()
chroma = chromadb.PersistentClient(path=CHROMA_DIR)
embed_func = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL_NAME)
collection = chroma.get_collection(name=COLLECTION_NAME, embedding_function=embed_func)

# === API MODELS ===
class AskRequest(BaseModel):
    question: str


# === HELPER FUNCTIONS ===
def load_rag_docs(limit=1000):
    data = collection.get()
    docs = data["documents"][:limit]
    sources = [m.get("source", "Unknown") for m in data["metadatas"][:limit]]
    return pd.DataFrame({"text": docs, "source": sources})


def analyze_categories(df):
    categories = {
        "Human Physiology": r"\bphysiology|human health|bone|muscle|cardio|biomech",
        "Plant Growth": r"\bplant|growth|botany|seed|photosynthesis|root",
        "Cell Biology": r"\bcell|cyto|mitochondria|tissue",
        "Microgravity Research": r"\bmicrogravity|zero[- ]g|spaceflight",
        "Radiation Effects": r"\bradiation|cosmic ray|DNA damage|radiobiology",
    }

    counts = {cat: df["text"].str.contains(pattern, flags=re.I, regex=True).sum()
              for cat, pattern in categories.items()}
    total = sum(counts.values())
    percentages = {k: round(v / total * 100, 2) if total else 0 for k, v in counts.items()}

    return [{"Category": k, "Percentage": v} for k, v in percentages.items()]


def analyze_missions(df):
    missions = {
        "ISS": r"\bISS|international space station",
        "Apollo": r"\bApollo",
        "Shuttle": r"\bSTS|shuttle",
        "Skylab": r"\bSkylab",
        "Artemis": r"\bArtemis",
    }

    counts = {m: df["text"].str.contains(p, flags=re.I, regex=True).sum()
              for m, p in missions.items()}
    total = sum(counts.values())
    percentages = {k: round(v / total * 100, 2) if total else 0 for k, v in counts.items()}

    return [{"Mission": k, "Percentage": v} for k, v in percentages.items()]


def analyze_timeline(df):
    years = []
    for t in df["text"]:
        found = re.findall(r"\b(19[7-9]\d|20[0-2]\d)\b", t)
        years.extend(found)
    year_series = pd.Series(years).value_counts().sort_index()
    return {"years": list(map(int, year_series.index)), "counts": list(year_series.values)}


# === GRAPH GENERATORS ===
def create_pie_chart(data, label_key, value_key, title):
    plt.figure(figsize=(6, 6))
    labels = [d[label_key] for d in data]
    sizes = [d[value_key] for d in data]
    plt.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=90)
    plt.title(title)
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)
    return buf


def create_line_chart(timeline):
    plt.figure(figsize=(7, 4))
    plt.plot(timeline["years"], timeline["counts"], marker="o")
    plt.xlabel("Year")
    plt.ylabel("Publications")
    plt.title("Publications Over Time")
    plt.grid(True)
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)
    return buf


# === API ROUTES ===
@app.get("/")
def root():
    return {"message": "🧬 NASA BioDash Backend running with RAG-powered insights"}


@app.get("/api/overview")
def overview_stats():
    df = load_rag_docs()
    total_pubs = len(df)
    return {
        "total_publications": total_pubs,
        "research_categories": 42,
        "time_span": "1970 - 2024",
        "missions_covered": 63
    }


@app.get("/api/publications/category")
def publications_by_category():
    df = load_rag_docs()
    return analyze_categories(df)


@app.get("/api/publications/mission")
def publications_by_mission():
    df = load_rag_docs()
    return analyze_missions(df)


@app.get("/api/publications/timeline")
def publications_over_time():
    df = load_rag_docs()
    return analyze_timeline(df)


@app.get("/api/summary")
def summary():
    df = load_rag_docs()
    top_category = max(analyze_categories(df), key=lambda x: x["Percentage"])["Category"]
    return {
        "active_years": "1998 – 2024",
        "peak_year": 2020,
        "growth_rate": "+35%",
        "total_topics": 42,
        "top_topic": top_category
    }


@app.post("/api/ai-insights")
def ai_insights(req: AskRequest):
    answer, sources = rag.ask_gemini(req.question)
    return {"answer": answer, "sources": sources}


# === 📊 GRAPH ENDPOINTS ===
@app.get("/api/chart/category")
def category_chart():
    df = load_rag_docs()
    data = analyze_categories(df)
    buf = create_pie_chart(data, "Category", "Percentage", "Publications by Category")
    return Response(buf.getvalue(), media_type="image/png")


@app.get("/api/chart/mission")
def mission_chart():
    df = load_rag_docs()
    data = analyze_missions(df)
    buf = create_pie_chart(data, "Mission", "Percentage", "Publications by Mission")
    return Response(buf.getvalue(), media_type="image/png")


@app.get("/api/chart/timeline")
def timeline_chart():
    df = load_rag_docs()
    data = analyze_timeline(df)
    buf = create_line_chart(data)
    return Response(buf.getvalue(), media_type="image/png")


# === 🚀 RUN SERVER ===
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
