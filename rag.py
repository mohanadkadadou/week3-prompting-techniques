"""RAG — policy-grounded answers.

Task: answer a customer question using ONLY the retrieved help-center text. If the
answer isn't in the docs, the model must say it doesn't know (don't hallucinate).

Fill in YOUR_SYSTEM_PROMPT and YOUR_RETRIEVER.
Run:  python rag.py
"""
import os

from ollama import chat

MODEL = "llama3.1:8b"
DATA = os.path.join(os.path.dirname(__file__), "data", "help_center.txt")


def load_chunks() -> list[str]:
    with open(DATA, encoding="utf-8") as f:
        # one chunk per non-empty line
        return [line.strip() for line in f if line.strip()]


# TODO: return the subset of chunks relevant to `question` (simple keyword overlap is fine).
STOPWORDS = {"a", "an", "the", "do", "does", "i", "you", "your", "my", "we", "is", "are", "to", "of",
             "how", "what", "what's", "long", "have", "can", "for", "it", "in", "on", "and", "or", "item"}


def _tokens(text: str) -> set[str]:
    words = {w.strip(".,:;?!'\"()").lower() for w in text.replace("-", " ").split()}
    # crude stemming so "ship", "ships", "shipping" and "internationally"/"international" overlap
    return {w[:6] for w in words if w and w not in STOPWORDS}


def YOUR_RETRIEVER(question: str, chunks: list[str]) -> list[str]:
    """Keyword-overlap retrieval: keep chunks sharing at least one (stemmed) content word, best first, top 3."""
    q = _tokens(question)
    scored = [(len(q & _tokens(c)), c) for c in chunks]
    ranked = sorted((s for s in scored if s[0] > 0), key=lambda s: -s[0])
    return [c for _, c in ranked[:3]]


# TODO: Fill this in! Force the model to answer only from the provided context.
YOUR_SYSTEM_PROMPT = """You are a support-desk assistant. Answer the customer's question using ONLY the
text under "Context". Do not use outside knowledge and do not guess.
If the Context is empty or does not contain the answer, reply exactly:
"Sorry, I don't know the answer to that from our help center."
Keep the answer to one or two sentences."""

CASES = [
    ("How long do I have to return an item?", ["30 days"]),
    ("Do you ship internationally?", ["international"]),
    ("What's your CEO's home address?", ["don't know", "do not know", "not sure", "cannot"]),
]


def answer(question: str) -> str:
    context = "\n".join(YOUR_RETRIEVER(question, load_chunks()))
    resp = chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": YOUR_SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ],
        options={"temperature": 0.0},
    )
    return resp.message.content.strip().lower()


def run() -> bool:
    passed = 0
    for question, needles in CASES:
        got = answer(question)
        ok = any(n.lower() in got for n in needles)
        passed += ok
        print(f"[{'PASS' if ok else 'FAIL'}] {question!r} -> {got[:80]!r}")
    print(f"{passed}/{len(CASES)} correct")
    return passed == len(CASES)


if __name__ == "__main__":
    run()
