import os
import sys
import requests
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

CSV_PATH = "company_database.csv"
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama-3.1-8b-instant"
TOP_K = 3         

API_KEY = "key"

if not API_KEY or "your_actual_key_here" in API_KEY:
    print("ERROR: Paste your Groq API key into the API_KEY variable at the top of this file.")
    print("Get one free at https://console.groq.com -> API Keys -> Create Key")
    sys.exit(1)

df = pd.read_csv(CSV_PATH)

def row_to_text(row):
    return (
        f"Company: {row['company_name']}. "
        f"Industry: {row['industry']}. "
        f"Skills required: {row['skills_required']}. "
        f"Experience needed: {row['experience']}. "
        f"Team size: {row['no_of_members']} members. "
        f"Location: {row['location']}."
    )

df["doc_text"] = df.apply(row_to_text, axis=1)

vectorizer = TfidfVectorizer(stop_words="english")
doc_vectors = vectorizer.fit_transform(df["doc_text"])

def retrieve_context(query, top_k=TOP_K):
    query_vec = vectorizer.transform([query])
    scores = cosine_similarity(query_vec, doc_vectors).flatten()
    top_idx = scores.argsort()[::-1][:top_k]
    matched_docs = df.iloc[top_idx]["doc_text"].tolist()
    return "\n".join(f"- {d}" for d in matched_docs)

def ask_groq(user_question, context, history):
    system_prompt = (
        "You are a helpful assistant that answers questions about companies "
        "using ONLY the context provided below. If the answer is not in the "
        "context, say you don't have that information.\n\n"
        f"Context:\n{context}"
    )

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history[-6:])  # keep last few turns for continuity
    messages.append({"role": "user", "content": user_question})

    payload = {
        "model": GROQ_MODEL,
        "messages": messages,
        "temperature": 0.3,
        "max_tokens": 512,
    }
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

    response = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=30)

    if response.status_code != 200:
        return f"[Groq API error {response.status_code}] {response.text}"

    data = response.json()
    return data["choices"][0]["message"]["content"]

def main():
    print("Company Database Chatbot (Groq + llama-3.1-8b-instant)")
    print(f"Loaded {len(df)} company records from {CSV_PATH}")
    print("Type 'exit' to quit.\n")

    history = []

    while True:
        query = input("You: ").strip()
        if query.lower() in ("exit", "quit"):
            print("Bot: Goodbye!")
            break
        if not query:
            continue

        context = retrieve_context(query)
        answer = ask_groq(query, context, history)

        print(f"Bot: {answer}\n")

        history.append({"role": "user", "content": query})
        history.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()