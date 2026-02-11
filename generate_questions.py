"""
Generate 30 multiple-choice Q/A pairs from the US Federal Budget FY2025.

Strategy:
  1. Load the budget text and split it into pages
  2. Sample 30 chunks from different parts of the document
  3. For each chunk, ask an LLM to generate a question + 4 choices + correct answer
  4. Save everything to questions.json
"""

import json
import random
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

# ---------------------------------------------------------------------------
# 1. Load the budget text and split into pages
# ---------------------------------------------------------------------------

with open("budget_fy2025.txt", "r") as f:
    full_text = f.read()

pages = full_text.split("--- PAGE ")
pages = [p.strip() for p in pages if p.strip()]  # drop empties

print(f"Loaded {len(pages)} pages from budget_fy2025.txt")

# ---------------------------------------------------------------------------
# 2. Sample 30 chunks spread across the document
# ---------------------------------------------------------------------------
# We want a good spread: some early (overview), some middle (departments),
# some late (appendix/tables). Skip the first few pages (title, TOC, legalese).

usable_pages = pages[4:]  # skip title/TOC/boilerplate
random.seed(42)

# Pick 30 pages spread evenly, with a bit of randomness
step = len(usable_pages) / 30
sampled = []
for i in range(30):
    center = int(i * step)
    # jitter by +/- 1 to avoid always picking the same position
    jitter = random.randint(-1, 1)
    idx = max(0, min(len(usable_pages) - 1, center + jitter))
    sampled.append(usable_pages[idx])

print(f"Sampled {len(sampled)} chunks for question generation")

# ---------------------------------------------------------------------------
# 3. For each chunk, generate a multiple-choice question
# ---------------------------------------------------------------------------

PROMPT_TEMPLATE = """You are creating a multiple-choice quiz about the US Federal Budget for Fiscal Year 2025.

Below is an excerpt from the budget document. Write ONE factual question that can be answered from this excerpt.

Rules:
- The question must be answerable ONLY from the text provided
- The question should ask about a specific fact, number, program, or policy
- Provide exactly 4 answer choices labeled A, B, C, D
- Exactly one choice must be correct
- The wrong choices should be plausible but clearly wrong based on the text
- Vary the position of the correct answer (don't always make it A)

Respond in this exact JSON format (no markdown, no code fences):
{"question": "...", "choices": {"A": "...", "B": "...", "C": "...", "D": "..."}, "correct": "A", "explanation": "Brief explanation of why this is correct"}

EXCERPT:
"""

questions = []

for i, chunk in enumerate(sampled):
    print(f"Generating question {i+1}/30...", end=" ", flush=True)

    prompt = PROMPT_TEMPLATE + chunk[:3000]

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )

    raw = response.choices[0].message.content.strip()

    # Parse the JSON response and shuffle choices so the correct answer
    # isn't always in the same position (LLMs love putting it at A)
    try:
        q = json.loads(raw)
        correct_text = q["choices"][q["correct"]]
        items = list(q["choices"].values())
        random.shuffle(items)
        labels = ["A", "B", "C", "D"]
        q["choices"] = {l: v for l, v in zip(labels, items)}
        q["correct"] = labels[items.index(correct_text)]
        q["source_chunk_preview"] = chunk[:200]
        q["chunk_index"] = i
        questions.append(q)
        print(f"OK [{q['correct']}]  ->  {q['question'][:60]}...")
    except (json.JSONDecodeError, KeyError):
        print(f"SKIP (bad JSON)")
        continue

# ---------------------------------------------------------------------------
# 4. Save to questions.json
# ---------------------------------------------------------------------------

with open("questions.json", "w") as f:
    json.dump(questions, f, indent=2)

print(f"\nDone! Saved {len(questions)} questions to questions.json")
