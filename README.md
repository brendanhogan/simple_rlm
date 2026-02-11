# Simple RLM

A simple, single-file implementation of **Recursive Language Models (RLMs)** based on [Zhang, Kraska, and Khattab (MIT CSAIL, 2025)](https://arxiv.org/abs/2512.24601).

The core idea: instead of stuffing a huge document into an LLM's context window, load it as a variable in a simulated Python REPL and let the LLM write code to explore it — searching for keywords, reading chunks, and optionally making recursive sub-LLM calls to reason about what it finds.

## What's in here

| File | What it does |
|---|---|
| `run_rlm.py` | The main script — runs Baseline vs RLM on 30 questions and prints results |
| `generate_questions.py` | Generates the 30 multiple-choice questions from the budget document |
| `questions.json` | The 30 pre-generated questions |
| `budget_fy2025.txt` | US Federal Budget FY2025 (~612K chars, ~150K tokens) |

## Setup

```bash
pip install openai python-dotenv
```

Create a `.env` file with your OpenAI API key:

```
OPENAI_API_KEY=sk-...
```

## Usage

```bash
# Run all 30 questions (baseline + RLM for each)
python run_rlm.py

# Run just the first N questions (useful for testing)
python run_rlm.py -n 1
python run_rlm.py -n 5
```

## How it works

**Baseline** — Truncates the document to fit the context window, sends it with the question, asks for an answer. Simple but limited.

**RLM** — Loads the full document into a Python variable `context`, gives the LLM a REPL with:
- `context` — the full document as a string
- `llm_query(prompt)` — call a sub-LLM to reason about text chunks
- `print()` — see output from code execution

The LLM writes code to search, zoom in, and answer. Each step is printed to the terminal so you can watch it think. A detailed log file (`rlm_log_<timestamp>.txt`) is written at the end with the full conversation trace for every question.

## Example output

```
============================================================
  Question 1/30
  Q: How many significant bipartisan bills supporting veterans...?
  A) 30   B) 50   C) 20   D) 10
============================================================

  -- Baseline --
  Answer: A  [+]    Time: 1.3s

  -- RLM --
  Step 1 | LLM writes code:
    | for i, line in enumerate(context.split('\n')):
    |     if 'bipartisan' in line.lower() and 'veteran' in line.lower():
    |         print(f"Line {i}: {line[:150]}")
  Step 1 | Output:
    | Line 204: service. Since I took office, I have signed over 30...

  >> FINAL ANSWER: A    Time: 1.9s    (2 steps, 0 sub-calls)
```

## Models

Configured at the top of `run_rlm.py`:
- **Root model**: `gpt-4.1` (both baseline and RLM root)
- **Sub model**: `gpt-4.1` (recursive calls from within the REPL)

## References

- [Recursive Language Models (Zhang et al., 2025)](https://arxiv.org/abs/2512.24601)
