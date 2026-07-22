import PyPDF2
from groq import Groq

# ==========================
# GROQ API KEY
# ==========================
GROQ_API_KEY = "YOUR API KEY"

client = Groq(
    api_key=GROQ_API_KEY
)

# ==========================
# PDF FILE
# ==========================
BOOK_FILE = "book1.pdf"

# ==========================
# READ PDF
# ==========================
def read_pdf(file):
    text = ""

    with open(file, "rb") as f:
        reader = PyPDF2.PdfReader(f)

        for page in reader.pages:
            page_text = page.extract_text()

            if page_text:
                text += page_text + "\n"

    return text

book_content = read_pdf(BOOK_FILE)

print("✅ PDF Loaded Successfully!")
print("Type 'exit' to quit.")

# ==========================
# CHAT LOOP
# ==========================
while True:

    question = input("\nYou: ")

    if question.lower() == "exit":
        print("Bot: Goodbye!")
        break

    prompt = f"""
You are a helpful PDF assistant.

Answer ONLY from the PDF content below.

If the answer is not present in the PDF,
reply exactly:

Not in PDF

PDF CONTENT:
{book_content[:4000]}

QUESTION:
{question}
"""

    try:

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.2,
            max_tokens=200
        )

        answer = response.choices[0].message.content

        print("\nBot:", answer)

    except Exception as e:

        print("\nError:", str(e))