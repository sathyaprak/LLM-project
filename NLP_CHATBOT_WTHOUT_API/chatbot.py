"""
Rule-Based NLP Chatbot (NO AI / NO Machine Learning)
=====================================================
This chatbot works purely through:
  1. Text cleaning (lowercasing, punctuation removal)
  2. Keyword / substring matching against a JSON "intents" dataset
  3. Simple scoring (most matched words wins) to pick the best intent
  4. Random response selection from that intent's response list

No neural networks, no scikit-learn, no embeddings, no training —
just string processing and if/else-style logic.

Files:
  - intents.json   -> the "dataset" (tags, patterns, responses)
  - chatbot.py     -> this file, the matching engine + chat loop

Run:
  python chatbot.py
"""

import json
import random
import re
import string
from datetime import datetime

INTENTS_FILE = "intents.json"


# ---------------------------------------------------------
# 1. Load the dataset
# ---------------------------------------------------------
def load_intents(path=INTENTS_FILE):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["intents"]


# ---------------------------------------------------------
# 2. Text cleaning
# ---------------------------------------------------------
def clean_text(text):
    text = text.lower().strip()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text)
    return text


def tokenize(text):
    return clean_text(text).split()


# ---------------------------------------------------------
# 3. Matching logic: score each intent by keyword overlap
# ---------------------------------------------------------
def score_intent(user_tokens, pattern):
    """Count how many words in the pattern appear in the user's message.
    Also gives a bonus for an exact full-phrase substring match."""
    pattern_clean = clean_text(pattern)
    pattern_tokens = pattern_clean.split()

    overlap = sum(1 for word in pattern_tokens if word in user_tokens)

    score = overlap
    if pattern_clean in " ".join(user_tokens):
        score += 2  # bonus for exact phrase match

    return score


def find_best_intent(user_input, intents, threshold=1):
    user_tokens = tokenize(user_input)
    best_tag = None
    best_score = 0

    for intent in intents:
        if intent["tag"] == "fallback":
            continue
        for pattern in intent["patterns"]:
            s = score_intent(user_tokens, pattern)
            if s > best_score:
                best_score = s
                best_tag = intent["tag"]

    if best_score < threshold:
        return "fallback"
    return best_tag


# ---------------------------------------------------------
# 4. Response generation
# ---------------------------------------------------------
def get_response(tag, intents):
    for intent in intents:
        if intent["tag"] == tag:
            response = random.choice(intent["responses"])

            # Handle dynamic placeholders
            if response == "__TIME__":
                return f"The current time is {datetime.now().strftime('%H:%M:%S')}."
            if response == "__DATE__":
                return f"Today's date is {datetime.now().strftime('%Y-%m-%d')}."

            return response

    return "I'm not sure how to respond to that."


# ---------------------------------------------------------
# 5. Chat loop
# ---------------------------------------------------------
def chat():
    intents = load_intents()
    print("RuleBot: Hi! I'm a rule-based chatbot (no AI). Type 'quit' to exit.\n")

    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if clean_text(user_input) in ("quit", "exit", "stop"):
            print("RuleBot: Goodbye!")
            break

        tag = find_best_intent(user_input, intents)
        response = get_response(tag, intents)
        print(f"RuleBot: {response}")


# ---------------------------------------------------------
# 6. Optional: quick automated test (no user input required)
# ---------------------------------------------------------
def run_demo():
    intents = load_intents()
    test_messages = [
        "hi there",
        "what is your name",
        "tell me a joke",
        "what time is it",
        "thanks a lot",
        "asdkjaslkdj random gibberish",
        "bye",
    ]
    print("--- Demo run (no input needed) ---")
    for msg in test_messages:
        tag = find_best_intent(msg, intents)
        response = get_response(tag, intents)
        print(f"You: {msg}\nRuleBot [{tag}]: {response}\n")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        run_demo()
    else:
        chat()
