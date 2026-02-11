"""
run_rlm.py — Compare Baseline vs RLM on 30 budget questions.

Baseline: stuff the full document + question into one prompt.
RLM: give the LLM a REPL with the document loaded as a variable,
     let it write code to explore, and optionally make sub-LLM calls.

Produces a detailed log file (rlm_log_<timestamp>.txt) with the full
conversation trace for every RLM question — system prompt, each code
block, each output, every sub-LLM call with full prompt and response.
"""

import argparse, ast, json, os, re, sys, time, io
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

# ─── Configuration ────────────────────────────────────────────────────────────
ROOT_MODEL = "gpt-4.1"       # used for baseline and RLM root
SUB_MODEL  = "gpt-4.1"       # used for sub-LLM calls inside the REPL
MAX_STEPS  = 10               # max REPL iterations before we force-stop

# gpt-4o-mini has a 128K token context window (~512K chars).
# Our document is ~612K chars, so baseline must truncate.
# This is exactly the limitation RLM is designed to overcome.
BASELINE_MAX_CHARS = 400_000  # leave room for prompt + response tokens

DOCUMENT_PATH  = "budget_fy2025.txt"
QUESTIONS_PATH = "questions.json"


# ═══════════════════════════════════════════════════════════════════════════════
#  Log file — collects detailed RLM traces, written at the end
# ═══════════════════════════════════════════════════════════════════════════════

_log_lines = []

def log(text=""):
    """Append a line to the log file buffer."""
    _log_lines.append(text)

def log_section(title):
    log(f"\n{'─' * 70}")
    log(f"  {title}")
    log(f"{'─' * 70}")

def log_block(label, content, max_chars=None):
    """Log a labeled block of text, optionally truncated."""
    log(f"\n  [{label}]")
    if max_chars and len(content) > max_chars:
        content = content[:max_chars] + f"\n... (truncated, {len(content):,} chars total)"
    for line in content.split("\n"):
        log(f"    {line}")

def write_log_file():
    """Write the accumulated log to a timestamped file."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(script_dir, f"rlm_log_{timestamp}.txt")
    with open(log_path, "w") as f:
        f.write("\n".join(_log_lines))
    return log_path


# ═══════════════════════════════════════════════════════════════════════════════
#  Section 1: LLM helpers
# ═══════════════════════════════════════════════════════════════════════════════

def ask_llm(prompt, model=ROOT_MODEL, system=None):
    """Send a simple prompt to the LLM and return the text response."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    response = client.chat.completions.create(model=model, messages=messages, temperature=0)
    return response.choices[0].message.content.strip()


def ask_llm_chat(messages, model=ROOT_MODEL):
    """Send a multi-turn conversation and return the assistant's text response."""
    response = client.chat.completions.create(model=model, messages=messages, temperature=0)
    return response.choices[0].message.content.strip()


# ═══════════════════════════════════════════════════════════════════════════════
#  Section 2: The two approaches
# ═══════════════════════════════════════════════════════════════════════════════

# ─── Baseline ─────────────────────────────────────────────────────────────────

def run_baseline(question, choices, document):
    """Stuff the document + question into one prompt. Truncates if needed. Return (answer, seconds)."""
    choices_str = "\n".join(f"  {k}) {v}" for k, v in choices.items())

    doc = document
    if len(doc) > BASELINE_MAX_CHARS:
        doc = doc[:BASELINE_MAX_CHARS] + "\n\n[... document truncated ...]"

    prompt = f"""You are answering a multiple-choice question about the following document.

--- DOCUMENT START ---
{doc}
--- DOCUMENT END ---

Question: {question}
{choices_str}

Respond with ONLY the letter (A, B, C, or D). Nothing else."""

    t0 = time.time()
    answer = ask_llm(prompt)
    elapsed = time.time() - t0

    letter = extract_letter(answer)
    return letter, elapsed


# ─── RLM ──────────────────────────────────────────────────────────────────────

RLM_SYSTEM_PROMPT = """You are solving a multiple-choice question about a large document ({doc_len:,} characters).

You have a Python REPL. Variables and functions available:
  - `context`  — the full document as a string
  - `llm_query(prompt)` — call a language model, returns string response
  - `print()` — use this to see output (ALWAYS use print, bare expressions won't show)

How to respond:
  - To run code: write a ```repl code block. You'll see the printed output.
  - To answer:   write FINAL(X) where X is A, B, C, or D.

Strategy:
  1. Search for keywords: print lines containing relevant terms
  2. Read nearby context: extract a chunk around the matching lines
  3. If needed, call llm_query() with the chunk + a focused question
  4. Answer with FINAL(X)

Example:
```repl
# Step 1: find relevant lines
for i, line in enumerate(context.split('\\n')):
    if 'keyword' in line.lower():
        print(f"Line {{i}}: {{line[:120]}}")
```

RULES:
  - ALWAYS use print() to see results. Bare expressions produce no output.
  - Do NOT print the entire document.
  - Be efficient. Usually 2-4 steps is enough.
"""

def run_rlm(question, choices, document, question_num=0, total=0):
    """
    Give the LLM a REPL with the document loaded.
    Let it write code to explore, make sub-LLM calls, and answer.
    Returns (answer_letter, elapsed_seconds, num_steps, num_sub_calls).
    """
    choices_str = "\n".join(f"  {k}) {v}" for k, v in choices.items())
    sub_call_count = 0
    sub_call_log = []  # collect full sub-LLM details for the log

    # ── Log: question header ──
    log(f"\n{'═' * 70}")
    log(f"  RLM TRACE — Question {question_num}/{total}")
    log(f"  Q: {question}")
    log(f"  Choices: {choices_str}")
    log(f"{'═' * 70}")

    # Build the REPL namespace
    def llm_query(prompt):
        nonlocal sub_call_count
        sub_call_count += 1
        print_sub_call(prompt, sub_call_count)

        # Log the full sub-LLM prompt
        log(f"\n  ┌─ Sub-LLM call #{sub_call_count} ─────────────────────────────")
        log_block(f"Sub-LLM #{sub_call_count} FULL PROMPT", prompt, max_chars=2000)

        result = ask_llm(prompt, model=SUB_MODEL)

        # Log the full sub-LLM response
        log_block(f"Sub-LLM #{sub_call_count} FULL RESPONSE", result, max_chars=2000)
        log(f"  └─────────────────────────────────────────────────")

        print_sub_response(result, sub_call_count)
        return result

    namespace = {"context": document, "llm_query": llm_query}

    system = RLM_SYSTEM_PROMPT.format(doc_len=len(document))

    # Log the system prompt
    log_block("SYSTEM PROMPT", system)

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Question: {question}\n{choices_str}\n\nExplore the document and find the answer."},
    ]

    log_block("INITIAL USER MESSAGE", messages[-1]["content"])

    t0 = time.time()

    for step in range(1, MAX_STEPS + 1):
        response = ask_llm_chat(messages, model=ROOT_MODEL)
        messages.append({"role": "assistant", "content": response})

        # Log the full raw LLM response
        log_section(f"Step {step} — LLM Response")
        log_block(f"RAW LLM RESPONSE (step {step})", response, max_chars=3000)

        # Check for FINAL answer
        final = extract_final(response)
        if final:
            elapsed = time.time() - t0
            log(f"\n  >>> FINAL ANSWER: {final}    Time: {elapsed:.1f}s    ({step} steps, {sub_call_count} sub-calls)")
            print_rlm_final(final, elapsed, step, sub_call_count)
            return final, elapsed, step, sub_call_count

        # Extract and run code block
        code = extract_code_block(response)
        if not code:
            nudge = "Please write a ```repl``` code block to explore the document, or respond with FINAL(X) if you know the answer."
            messages.append({"role": "user", "content": nudge})
            log(f"\n  (no code block found — nudging LLM)")
            print_rlm_step(step, "(no code produced, nudging...)", "")
            continue

        print_rlm_code(step, code)
        log_block(f"CODE (step {step})", code)

        # Execute the code, capturing stdout
        stdout_capture = io.StringIO()
        namespace["print"] = lambda *args, **kwargs: print(*args, file=stdout_capture, **kwargs)

        try:
            tree = ast.parse(code)
            if tree.body and isinstance(tree.body[-1], ast.Expr):
                last_expr = tree.body.pop()
                if tree.body:
                    exec(compile(ast.Module(body=tree.body, type_ignores=[]), "<repl>", "exec"), namespace)
                result = eval(compile(ast.Expression(body=last_expr.value), "<repl>", "eval"), namespace)
                if result is not None:
                    stdout_capture.write(repr(result) + "\n")
            else:
                exec(code, namespace)
        except Exception as e:
            stdout_capture.write(f"ERROR: {e}\n")

        output = stdout_capture.getvalue()
        if len(output) > 3000:
            output = output[:3000] + "\n... (truncated)"

        log_block(f"OUTPUT (step {step})", output, max_chars=3000)

        print_rlm_output(step, output)

        user_msg = f"Output:\n{output}\n\nContinue exploring or respond with FINAL(X)."
        messages.append({"role": "user", "content": user_msg})

    # Ran out of steps
    elapsed = time.time() - t0
    messages.append({"role": "user", "content": "You've used all your steps. Please respond with FINAL(X) now with your best guess."})
    response = ask_llm_chat(messages, model=ROOT_MODEL)
    log_block("FORCED FINAL RESPONSE", response)
    final = extract_final(response) or extract_letter(response)
    log(f"\n  >>> FINAL ANSWER (forced): {final or '?'}    Time: {elapsed:.1f}s    ({MAX_STEPS} steps, {sub_call_count} sub-calls)")
    print_rlm_final(final or "?", elapsed, MAX_STEPS, sub_call_count)
    return final, elapsed, MAX_STEPS, sub_call_count


# ═══════════════════════════════════════════════════════════════════════════════
#  Parsing helpers
# ═══════════════════════════════════════════════════════════════════════════════

def extract_letter(text):
    """Pull a single A/B/C/D from the LLM response."""
    m = re.search(r'\b([A-D])\b', text)
    return m.group(1) if m else None


def extract_final(text):
    """Look for FINAL(X) in the response."""
    m = re.search(r'FINAL\(([A-D])\)', text)
    return m.group(1) if m else None


def extract_code_block(text):
    """Extract code from a ```repl ... ``` block (or any ``` block)."""
    m = re.search(r'```(?:repl|python)?\s*\n(.*?)```', text, re.DOTALL)
    return m.group(1).strip() if m else None


# ═══════════════════════════════════════════════════════════════════════════════
#  Pretty printing (terminal output)
# ═══════════════════════════════════════════════════════════════════════════════

def indent(text, prefix="    | "):
    """Indent each line of text with a prefix."""
    return "\n".join(prefix + line for line in text.split("\n"))


def print_question_header(i, total, question, choices):
    choices_str = "   ".join(f"{k}) {v}" for k, v in choices.items())
    print(f"\n{'=' * 60}")
    print(f"  Question {i}/{total}")
    print(f"  Q: {question}")
    print(f"  {choices_str}")
    print(f"{'=' * 60}")


def print_baseline_result(answer, elapsed, correct):
    mark = "+" if answer == correct else "x"
    print(f"\n  -- Baseline --")
    print(f"  Answer: {answer}  [{mark}]    Time: {elapsed:.1f}s")


def print_rlm_code(step, code):
    print(f"\n  Step {step} | LLM writes code:")
    print(indent(code))


def print_rlm_output(step, output):
    if output.strip():
        print(f"  Step {step} | Output:")
        print(indent(output))


def print_rlm_step(step, code, output):
    print(f"  Step {step} | {code}")


def print_sub_call(prompt, call_num):
    short = prompt[:120].replace("\n", " ")
    print(f"  >> Sub-LLM call #{call_num}: {short}...")


def print_sub_response(response, call_num):
    short = response[:120].replace("\n", " ")
    print(f"  << Sub-LLM #{call_num} response: {short}...")


def print_rlm_final(answer, elapsed, steps, sub_calls):
    print(f"\n  >> FINAL ANSWER: {answer}    Time: {elapsed:.1f}s    ({steps} steps, {sub_calls} sub-calls)")


def print_results_table(baseline_results, rlm_results, questions):
    n = len(questions)
    b_correct = sum(1 for i in range(n) if baseline_results[i][0] == questions[i]["correct"])
    r_correct = sum(1 for i in range(n) if rlm_results[i][0] == questions[i]["correct"])
    b_avg_time = sum(r[1] for r in baseline_results) / n
    r_avg_time = sum(r[1] for r in rlm_results) / n

    print(f"\n{'=' * 50}")
    print(f"  {'RESULTS':^46}")
    print(f"{'=' * 50}")
    print(f"  {'':20} {'Baseline':>12} {'RLM':>12}")
    print(f"  {'-' * 46}")
    print(f"  {'Accuracy':20} {b_correct}/{n} ({100*b_correct/n:.0f}%){'':<4} {r_correct}/{n} ({100*r_correct/n:.0f}%)")
    print(f"  {'Avg time':20} {b_avg_time:>10.1f}s {r_avg_time:>10.1f}s")
    print(f"{'=' * 50}")

    # Per-question breakdown
    print(f"\n  {'#':>3}  {'Correct':>7}  {'Baseline':>8}  {'RLM':>8}  {'B time':>7}  {'R time':>7}")
    print(f"  {'-' * 48}")
    for i in range(n):
        correct = questions[i]["correct"]
        b_ans = baseline_results[i][0] or "?"
        r_ans = rlm_results[i][0] or "?"
        b_mark = "+" if b_ans == correct else "x"
        r_mark = "+" if r_ans == correct else "x"
        b_t = baseline_results[i][1]
        r_t = rlm_results[i][1]
        print(f"  {i+1:>3}  {correct:>7}  {b_ans:>5} [{b_mark}]  {r_ans:>5} [{r_mark}]  {b_t:>6.1f}s  {r_t:>6.1f}s")


# ═══════════════════════════════════════════════════════════════════════════════
#  Section 3: Main — run all questions and print results
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Compare Baseline vs RLM on budget questions.")
    parser.add_argument("-n", "--num-questions", type=int, default=None,
                        help="Number of questions to run (default: all)")
    args = parser.parse_args()

    # Load data
    script_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(script_dir, DOCUMENT_PATH), "r") as f:
        document = f.read()
    with open(os.path.join(script_dir, QUESTIONS_PATH), "r") as f:
        questions = json.load(f)

    if args.num_questions is not None:
        questions = questions[:args.num_questions]

    print(f"\nLoaded document: {len(document):,} characters")
    print(f"Running {len(questions)} question(s)")
    print(f"Models — root: {ROOT_MODEL}, sub: {SUB_MODEL}")
    if len(document) > BASELINE_MAX_CHARS:
        print(f"Note: baseline will truncate document to {BASELINE_MAX_CHARS:,} chars (context window limit)")
        print(f"      RLM explores the full {len(document):,} char document via code")

    # Log header
    log(f"RLM Detailed Log")
    log(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"Root model: {ROOT_MODEL}    Sub model: {SUB_MODEL}")
    log(f"Document: {len(document):,} chars    Questions: {len(questions)}")
    log(f"Max steps per question: {MAX_STEPS}")

    baseline_results = []
    rlm_results = []

    for i, q in enumerate(questions, 1):
        question = q["question"]
        choices = q["choices"]

        print_question_header(i, len(questions), question, choices)

        # Run baseline
        b_answer, b_time = run_baseline(question, choices, document)
        print_baseline_result(b_answer, b_time, q["correct"])
        baseline_results.append((b_answer, b_time))

        # Log baseline result
        log(f"\n  Baseline: {b_answer} [{'correct' if b_answer == q['correct'] else 'wrong'}]  Time: {b_time:.1f}s")

        # Run RLM (logging happens inside run_rlm)
        print(f"\n  -- RLM --")
        r_answer, r_time, r_steps, r_subs = run_rlm(
            question, choices, document, question_num=i, total=len(questions)
        )
        rlm_results.append((r_answer, r_time, r_steps, r_subs))

        # Log RLM correctness
        log(f"  RLM correct: {'yes' if r_answer == q['correct'] else 'no'} (expected {q['correct']}, got {r_answer})")
        
    # Print summary
    print_results_table(baseline_results, rlm_results, questions)

    # Also log the summary table
    n = len(questions)
    b_correct = sum(1 for i in range(n) if baseline_results[i][0] == questions[i]["correct"])
    r_correct = sum(1 for i in range(n) if rlm_results[i][0] == questions[i]["correct"])
    log(f"\n{'═' * 70}")
    log(f"  SUMMARY")
    log(f"{'═' * 70}")
    log(f"  Baseline accuracy: {b_correct}/{n} ({100*b_correct/n:.0f}%)")
    log(f"  RLM accuracy:      {r_correct}/{n} ({100*r_correct/n:.0f}%)")

    # Write the log file
    log_path = write_log_file()
    print(f"\nDetailed log written to: {log_path}")


if __name__ == "__main__":
    main()
