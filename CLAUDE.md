# Simple RLM - Project Principles

## Core Philosophy
This is a simple, intuitive implementation of Recursive Language Models (RLMs) based on the paper by Zhang, Kraska, and Khattab (MIT CSAIL, 2025).

## Non-Negotiable Priorities (in order)
1. **Clarity over cleverness** — Every piece of code should be immediately understandable. If you have to think hard to read it, rewrite it.
2. **Easy to visualize** — The flow of what's happening (prompt → environment → peek/chunk → sub-calls → answer) should be obvious at every step. Logging and output should make the recursive process *visible*.
3. **Intuitive structure** — A new reader should be able to open the code and understand the full system in under 10 minutes.
4. **Minimal abstractions** — Don't wrap things in classes or patterns unless they genuinely help understanding. Flat is better than nested. Simple functions over complex architectures.
5. **Correctness** — It should actually work and demonstrate the core RLM insight clearly.

## What This Is NOT
- Not a production framework
- Not a benchmark reproduction
- Not optimized for performance or cost
- Not trying to be general-purpose

## What This IS
- A clear, visual demonstration of the RLM idea
- Code that reads almost like pseudocode
- Something you could walk someone through in a conversation
