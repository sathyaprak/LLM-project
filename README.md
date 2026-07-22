# 🏢 Company Database Chatbot

An AI-powered Company Database Chatbot built with **Python**, **Groq Llama 3.1 8B Instant**, and **TF-IDF Retrieval**. The chatbot retrieves relevant company information from a CSV database using semantic similarity and generates intelligent, context-aware responses with the Groq API.

---

## 📌 Overview

This project is a Retrieval-Augmented Generation (RAG)-style chatbot that enables users to ask natural language questions about companies stored in a CSV database.

Instead of sending the entire dataset to the LLM, the chatbot retrieves the most relevant company records using **TF-IDF Vectorization** and **Cosine Similarity**, then provides the retrieved context to **Groq's Llama 3.1 8B Instant** model to generate accurate responses.

---

## ✨ Features

- 🤖 AI-powered chatbot using Groq Llama 3.1
- 📊 Company information stored in CSV
- 🔍 TF-IDF based semantic search
- 📈 Cosine Similarity retrieval
- ⚡ Fast inference using Groq API
- 💬 Multi-turn conversation support
- 🧠 Context-aware responses
- 🛡️ Hallucination reduction using retrieved context
- 🖥️ Simple Command Line Interface
- 📂 Easy to customize with your own dataset

---

## 🛠️ Technologies Used

- Python 3.x
- Groq API
- Llama 3.1 8B Instant
- Pandas
- Scikit-learn
- TF-IDF Vectorizer
- Cosine Similarity
- Requests
- CSV Database

---

## 📂 Project Structure

```
Company-Database-Chatbot
│
├── company_database.csv
├── chatbot.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## ⚙️ Installation

### Clone Repository

```bash
git clone https://github.com/yourusername/company-database-chatbot.git

cd company-database-chatbot
```

### Create Virtual Environment

Windows

```bash
python -m venv venv
venv\Scripts\activate
```

Linux / macOS

```bash
python3 -m venv venv

source venv/bin/activate
```

---

### Install Dependencies

```bash
pip install -r requirements.txt
```

or

```bash
pip install pandas scikit-learn requests
```

---

## 🔑 Get Groq API Key

1. Visit

https://console.groq.com

2. Login

3. Create API Key

4. Replace

```python
API_KEY = "YOUR_API_KEY"
```

---

## ▶️ Run Project

```bash
python chatbot.py
```

Output

```
Company Database Chatbot (Groq + llama-3.1-8b-instant)

Loaded 100 company records

You:
```

---

## 💬 Example Questions

```
Which companies require Python?

Show AI companies.

Which companies are located in Chennai?

Companies requiring Machine Learning.

How many members are required?

Suggest companies for freshers.

Which company needs Java developers?

Companies requiring React.

Tell me about Infosys.

Best company for Data Science.
```

---

## 🧠 How It Works

### Step 1

Load company database

↓

### Step 2

Convert each company record into text

↓

### Step 3

Generate TF-IDF vectors

↓

### Step 4

User asks question

↓

### Step 5

Calculate Cosine Similarity

↓

### Step 6

Retrieve Top-K company records

↓

### Step 7

Send retrieved context to Groq LLM

↓

### Step 8

Generate AI response

---

## 🏗️ Architecture

```
User
 │
 ▼
Question
 │
 ▼
TF-IDF Vectorizer
 │
 ▼
Cosine Similarity Search
 │
 ▼
Top K Company Records
 │
 ▼
Groq API
 │
 ▼
Llama 3.1 8B Instant
 │
 ▼
AI Response
```

---

## 📊 Retrieval Method

The chatbot uses

- TF-IDF Vectorizer
- Cosine Similarity
- Top-K Retrieval

This improves response quality by only sending the most relevant company records to the LLM.

---

## 📈 Advantages

- Faster than sending the entire CSV
- Lower token usage
- Reduced hallucinations
- Better accuracy
- Easy to scale
- Lightweight implementation
- Supports large datasets

---

## 🔮 Future Improvements

- FAISS Vector Database
- Sentence Transformers
- Streamlit Web App
- Flask API
- FastAPI Backend
- PDF Support
- Voice Chat
- User Authentication
- Chat History Database
- Docker Support

---

## 📦 Requirements

```
Python >= 3.10

pandas

scikit-learn

requests
```

---

## 📄 requirements.txt

```
pandas
scikit-learn
requests
```

---

## 👨‍💻 Author

**Mohamed Hamdhan S**

Computer Science Engineering Student

AI | Machine Learning | Data Analytics | Python Developer

---

## ⭐ If you like this project

Give this repository a ⭐ on GitHub!

---

## 📜 License

This project is licensed under the MIT License.

---

## 🙏 Acknowledgements

- Groq
- Meta Llama 3.1
- Scikit-learn
- Pandas
- Python Community
