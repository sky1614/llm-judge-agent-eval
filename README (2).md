# LLM-as-Judge — B2B Outreach Message Evaluator

A standalone evaluator that scores AI-generated B2B outreach messages before they are sent. Part of a multi-agent lead-generation pipeline.

---

## The problem it solves

LLMs are good at generating outreach messages but poor at knowing when they've produced a generic, off-tone, or low-effort one. In a production pipeline that sends hundreds of cold emails, even a 10% rate of bad messages can tank deliverability and domain reputation.

This module adds a second LLM ("the judge") that evaluates each generated message on five dimensions before it leaves the pipeline. It acts as an automated QA gate — cheaper than a human reviewer, faster than A/B testing, and provably better than sending everything blind.

---

## Where it sits in the pipeline

```
Prospector → Scorer → Writer → [LLM Judge] → Delivery
                                    │
                                    ├─ score < 6.5 → regenerate once
                                    ├─ still < 6.5 → block, skip this lead
                                    ├─ score 6.5–9.0 → goes to human review queue
                                    └─ score ≥ 9.0 → auto-approve (no human needed)
```

The judge uses a **different, cheaper model** than the writer to avoid self-evaluation bias. In the production system this is `openai/gpt-4o-mini` judging messages written by `claude-sonnet`.

---

## Scoring dimensions

| Dimension | Weight | What it measures |
|---|---|---|
| personalization | 30% | Does the message reference the prospect's company, role, or a visible pain point — or is it generic copy-paste? |
| cultural_fit | 20% | Does tone, language, and references match the target market? |
| cta_strength | 20% | Is the call-to-action clear, low-friction, and time-bound? |
| tone_match | 15% | Does the tone match the campaign setting (professional / friendly / direct)? |
| clarity | 15% | Is the message easy to read in under 30 seconds? |

**Weighted score = Σ(dimension_score × weight)**

---

## How to run

```bash
pip install openai python-dotenv

export OPENROUTER_API_KEY=your_key_here   # get one at openrouter.ai/keys

python llm_judge.py
```

Without an API key the script prints the expected output structure so you can see the data shape.

**OpenRouter** is used because it gives a single API endpoint for many models, which is useful when you want to swap judge and generator models independently. Any OpenAI-compatible endpoint works — just change `base_url` in the code.

---

## Demo output (with API key)

```
============================================================
  SAMPLE 1 — Personalized message
============================================================
  Weighted score : 7.9 / 10
  Passed         : True  (threshold: 6.5)
  Auto-approve   : False  (threshold: 9.0)
  Latency        : 1243 ms

  Dimension breakdown:
    personalization    8.5/10  (weight 30%)  — Mentions company milestone and specific pain point.
    cultural_fit       8.0/10  (weight 20%)  — Appropriate formality for target market.
    cta_strength       8.0/10  (weight 20%)  — Specific day and time suggested.
    tone_match         7.5/10  (weight 15%)  — Professional tone matches campaign setting.
    clarity            7.0/10  (weight 15%)  — Concise but middle paragraph could be trimmed.

============================================================
  SAMPLE 2 — Generic message
============================================================
  Weighted score : 2.1 / 10
  Passed         : False  (threshold: 6.5)
  ...
```

---

## Real-world numbers (small sample — treat as directional)

From ~53 sent messages across 58 agent runs (as of mid-2026, dogfooding the system for our own B2B outreach):

- Average judge score: **7.2 / 10**
- Messages blocked (score < 6.5): **~8%** of generations
- Messages auto-approved (score ≥ 9.0): **~12%** of generations
- Messages rewritten once and then passing: **~5%** of generations

These numbers come from a small sample and should not be extrapolated. The value of the judge shows up most clearly in blocking the 8% — generic messages that the writer model produced when given thin prospect data.

---

## What this file is

This is a self-contained extract of the judge module from a production B2B lead-generation pipeline. The production version stores results in a PostgreSQL table (`JudgeEvaluationDB`) and integrates with a Celery + Redis task queue. Those dependencies are stripped here to make the logic portable and easy to read.

No real prospect data, API keys, company names, or credentials are included. All example messages use fictional companies.

---

## Files

```
llm_judge.py   — The evaluator (this is what you're here for)
README.md      — This file
```

---

## License

MIT
