from rag_helper import RAGRetriever

def main():
    print("🚀 NASA Research Assistant (Gemini v2 RAG Mode)")
    rag = RAGRetriever()

    while True:
        query = input("❓ Question: ").strip()
        if query.lower() in ["quit", "exit"]:
            break

        answer, sources = rag.ask_gemini(query)
        print("\n🤖 Answer:\n", answer)
        print("\n📚 Sources:")
        for src in sources:
            if isinstance(src, dict):
                print("-", src.get("source", "Unknown source"))
            else:
                print("-", src)
        print("-" * 60)

if __name__ == "__main__":
    main()
