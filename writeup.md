# Week 3 — Prompting Techniques: The Support-Desk Assistant

**Student:** Mohanad Kadadou (2309115218) · **Model:** `llama3.1:8b` via Ollama 0.40.6, temperature 0.0
**Rule followed:** only the `TODO` parts were edited (the prompts, the tool description, and the retriever in `rag.py`). The model and the checks were not touched, and no output is post-processed.
**Results:** all four checks pass: k-shot 4/4, chain-of-thought 3/3, tool calling PASS, RAG 3/3. Each file was run twice with identical results (temperature 0).

## 1. K-shot prompting (`k_shot_prompting.py`)
**Final prompt**
```text
You classify customer-support tickets for an online store.
Allowed labels: billing, shipping, returns, technical, other.
Reply with ONLY the label: one lowercase word, no punctuation, no explanation.

Message: I was charged $40 but my order total was $20.            -> billing
Message: Do you accept PayPal for payment?                        -> billing
Message: My tracking number hasn't updated in a week.             -> shipping
Message: The courier left my parcel at the wrong address.         -> shipping
Message: How do I send this jacket back? It doesn't fit.          -> returns
Message: I'd like my money back, the item arrived damaged.        -> returns
Message: The website shows an error when I try to pay.            -> technical
Message: I can't log in, the password reset email never arrives.  -> technical
Message: Do you have a store in Berlin?                           -> other
```
(In the file each label sits on its own line under its `Message:` line.)
**Sample:** `"The app crashes every time I open my cart."` → `technical`
**Design notes:** examples are different from the graded cases (no leakage), two per label, and include a tricky
boundary pair ("The website shows an error when I *pay*" is technical, not billing).
**What I learned:** the examples define the output *format* as much as the labels: a rule plus examples made the model answer with a bare lowercase word, which an instruction alone does not reliably do.

## 2. Chain of thought (`chain_of_thought.py`)
**Final prompt**
```text
You decide whether an order is eligible for a refund.
Think step by step, using this exact checklist, and write each step on its own line:
1. Final sale? If the item is final sale the answer is NO, stop.
2. Days since purchase: compare with the 30-day limit. More than 30 days means NO.
3. Condition: the item must be unused / in original condition. Used or opened means NO.
4. If every check passed the answer is YES.
Finish with a last line that contains ONLY `DECISION: YES` or `DECISION: NO`
(nothing after it, no punctuation, no extra text on that line).
```
**Sample input → output** (`"Bought 45 days ago, unused."`):
```text
1. Final sale? No, it's not a final sale.
2. Days since purchase: 45 days is more than 30 days, so NO.
3. Condition: the item is unused, so it's in original condition.
4. If every check passed the answer is YES.

DECISION: NO
```
**Observation:** the 8B model reached the right decision at step 2 but did not "stop": it still ran steps 3 and 4. The correct final line is what the grader reads, but the trace shows the model follows a checklist format better than a control-flow instruction.
**What I learned:** forcing a fixed, numbered reasoning scaffold puts the policy comparison in the text *before* the verdict, so the decision is derived from the facts instead of guessed, and a strict last-line format makes it machine-gradable.

## 3. Tool calling (`tool_calling.py`)
**Final prompt + tool description**
```text
System: You are a support-desk assistant for an online store.
You do not know any order's status. When a customer mentions an order number, you MUST call
get_order_status with that order id before answering, then reply briefly using the tool result.

Tool description: Look up the current status of a customer order by its order id. Call this whenever
the customer asks about, checks on, or tracks an order; never guess a status.
order_id: The order number exactly as the customer wrote it, digits only, e.g. "4412".
```
**Sample:** `"Hi, can you check on order 4412 for me?"` → `tool called with order_id='4412'` → `tool result: shipped, arriving Friday`
**What I learned:** the model never sees the function code, only the name, description and parameter text, so the description *is* the interface: telling it when to call (and what format the id has) is what made it emit a structured call instead of a chatty guess.

## 4. RAG (`rag.py`)
**Retriever** (keyword overlap, no libraries): lowercase, strip punctuation, drop stopwords, crude 6-character stemming so
`international` matches `international` and `return` matches `returns`/`returned`; rank chunks by number of shared words, keep the top 3, and return nothing when the overlap is 0.
**Final prompt**
```text
You are a support-desk assistant. Answer the customer's question using ONLY the
text under "Context". Do not use outside knowledge and do not guess.
If the Context is empty or does not contain the answer, reply exactly:
"Sorry, I don't know the answer to that from our help center."
Keep the answer to one or two sentences.
```
**Sample:** `"Do you ship internationally?"` → `yes, we offer international shipping to most countries. delivery takes 7-21 business days.`;
`"What's your CEO's home address?"` → retriever returns `[]` → `Sorry, I don't know the answer to that from our help center.`
**What I learned:** most of the "refuse when unknown" behavior comes from retrieval returning an empty context *and* an exact fallback sentence; the prompt alone is weak, retrieval quality decides whether the answer is grounded.

## Closing: why LLMs are good at structured tasks and where they need scaffolding
LLMs are strong at pattern completion, so a few examples, a checklist, or a schema is enough to make them
classify, route, and format reliably: the structure is in the prompt and the model continues it. They need
scaffolding wherever the answer depends on something they cannot know or cannot hold exactly: live data
(order status needs a tool), private facts (policies need retrieval), and multi-step logic (a small model skipped the "stop" instruction
and needed a rigid step list). In every case the engineering move was the same: put the missing information, the
required format, or the missing capability *into the context*, and then verify the output with a deterministic
check, which is also how coding agents are built around a model.

## Reproduce
```bash
brew install ollama && ollama serve &
ollama pull llama3.1:8b
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python ollama
for f in k_shot_prompting chain_of_thought tool_calling rag; do .venv/bin/python $f.py; done
```
