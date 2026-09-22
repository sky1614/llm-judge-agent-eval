"""
llm_judge.py — Standalone LLM-as-Judge evaluator for B2B outreach messages.

WHAT THIS DOES
--------------
Before any outreach email or WhatsApp message is sent, this module asks a
second LLM to score the message across five quality dimensions. If the score
is too low the message is rewritten (up to once). If it still fails it is
blocked entirely. If it scores above an auto-approve threshold it can be
sent without a human in the loop.

This pattern is sometimes called "Constitutional AI" for pipelines: you let
one model generate and a second model judge, avoiding self-evaluation bias
by using a cheaper/different model for the judge role.

PIPELINE POSITION
-----------------
Writer agent → [this module] → Delivery agent
                  ↓ score < PASS threshold: regenerate once
                  ↓ still fails: block message
                  ↓ score ≥ AUTO_APPROVE threshold: skip human review

SCORING DIMENSIONS (weighted, 0–10 scale each)
-----------------------------------------------
  personalization   30%   Does the message reference the prospect's
                          company, role, or a visible pain point — or is
                          it generic copy-paste?
  cultural_fit      20%   Does tone/language/references match the target
                          market? (e.g. formal Indian B2B vs casual SaaS)
  cta_strength      20%   Is the call-to-action clear, low-friction,
                          and time-bound? ("15-min call next week" beats
                          "let me know if interested")
  tone_match        15%   Does the tone match the campaign setting
                          (professional / friendly / direct)?
  clarity           15%   Is the message easy to read in under 30 seconds?
                          No jargon, no wall of text.

  weighted_score = sum(dimension_score × weight)

HOW TO RUN (standalone demo)
-----------------------------
  pip install openai python-dotenv

  export OPENROUTER_API_KEY=YOUR_API_KEY_HERE

  python llm_judge.py

  Output: JSON evaluation for two sample messages (one good, one bad).

REAL SYSTEM RESULTS (as of mid-2026)
--------------------------------------
  - Average judge score across ~53 sent messages: 7.2 / 10
  - Messages blocked by judge (score < 6.5): ~8% of generations
  - Messages auto-approved (score ≥ 9.0): ~12% of generations
  - Judge rewrites that passed on second attempt: ~5% of generations
  Note: these numbers come from a small sample (53 sends / 58 runs).
        Treat as directional, not statistically significant.
"""

import json
import os
import time
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Configuration — replace with your own values or set environment variables
# ---------------------------------------------------------------------------

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "YOUR_API_KEY_HERE")

# Use a DIFFERENT / CHEAPER model than your generation model to avoid
# self-evaluation bias. We use the fallback model here intentionally.
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "openai/gpt-4o-mini")  # fast + cheap
GENERATION_MODEL = os.getenv("GENERATION_MODEL", "anthropic/claude-sonnet-4-5")

# Score thresholds
MIN_PASS_SCORE = 6.5    # Below this → rewrite, then block if still failing
AUTO_APPROVE_SCORE = 9.0  # At or above this → skip human review queue
JUDGE_TIMEOUT_SECONDS = 8

# Scoring weights (must sum to 1.0)
DIMENSION_WEIGHTS = {
    "personalization": 0.30,
    "cultural_fit":    0.20,
    "cta_strength":    0.20,
    "tone_match":      0.15,
    "clarity":         0.15,
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class JudgeInput:
    """Everything the judge needs to evaluate a message."""
    message: str              # The outreach message to evaluate
    company_name: str         # Prospect's company (for personalization check)
    role: str                 # Prospect's role / title
    industry: str             # Prospect's industry
    channel: str              # "email" or "whatsapp"
    campaign_tone: str        # "professional", "friendly", or "direct"
    pain_points: list = field(default_factory=list)   # Known pain points


@dataclass
class DimensionScore:
    score: float   # 0–10
    reason: str    # One sentence from the judge


@dataclass
class JudgeResult:
    weighted_score: float
    passed: bool
    auto_approve_eligible: bool
    dimensions: dict   # {dimension_name: DimensionScore}
    judge_model: str
    latency_ms: int
    raw_response: str  # Preserved for debugging


# ---------------------------------------------------------------------------
# Core judge function
# ---------------------------------------------------------------------------

def evaluate_message(inp: JudgeInput) -> JudgeResult:
    """
    Send a message to the judge LLM and return a structured evaluation.

    Uses a strict JSON output format so the result is always parseable.
    Falls back to a zero score on any parsing or network error rather than
    crashing the pipeline — the caller should treat a zero score as a block.
    """
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("pip install openai")

    client = OpenAI(
        api_key=OPENROUTER_API_KEY,
        base_url="https://openrouter.ai/api/v1",
    )

    prompt = _build_judge_prompt(inp)

    t0 = time.time()
    try:
        response = client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=512,
            temperature=0.0,   # Deterministic scoring
            timeout=JUDGE_TIMEOUT_SECONDS,
            extra_headers={
                "HTTP-Referer": "https://example.com",   # Replace with your site
                "X-Title": "LLM Judge Demo",
            },
        )
        raw = response.choices[0].message.content or ""
    except Exception as e:
        # Network / timeout — fail safe: block the message
        return _error_result(str(e), JUDGE_MODEL, int((time.time() - t0) * 1000))

    latency_ms = int((time.time() - t0) * 1000)
    return _parse_judge_response(raw, JUDGE_MODEL, latency_ms)


def _build_judge_prompt(inp: JudgeInput) -> str:
    pain_str = ", ".join(inp.pain_points) if inp.pain_points else "unknown"
    weights_str = "\n".join(
        f"  - {dim} ({int(w*100)}%)" for dim, w in DIMENSION_WEIGHTS.items()
    )
    return f"""You are a senior B2B sales coach evaluating an outreach message.
Score this {inp.channel} message on each dimension from 0 to 10.

PROSPECT CONTEXT
  Company:    {inp.company_name}
  Role:       {inp.role}
  Industry:   {inp.industry}
  Pain points:{pain_str}
  Channel:    {inp.channel}
  Expected tone: {inp.campaign_tone}

MESSAGE TO EVALUATE
---
{inp.message}
---

SCORING DIMENSIONS (score 0–10 each, then compute weighted total)
{weights_str}

Scoring guide:
  personalization: 0 = completely generic, 10 = deeply tailored to this company/role
  cultural_fit:    0 = culturally off (wrong formality, odd references), 10 = perfect fit
  cta_strength:    0 = no clear ask, 10 = specific low-friction time-bound ask
  tone_match:      0 = wrong tone for channel/setting, 10 = exactly right
  clarity:         0 = confusing, jargon-heavy, wall of text, 10 = crisp and scannable

Return ONLY valid JSON, no markdown fences, no explanation outside the JSON:
{{
  "personalization": {{"score": <0-10>, "reason": "<one sentence>"}},
  "cultural_fit":    {{"score": <0-10>, "reason": "<one sentence>"}},
  "cta_strength":    {{"score": <0-10>, "reason": "<one sentence>"}},
  "tone_match":      {{"score": <0-10>, "reason": "<one sentence>"}},
  "clarity":         {{"score": <0-10>, "reason": "<one sentence>"}}
}}"""


def _parse_judge_response(raw: str, model: str, latency_ms: int) -> JudgeResult:
    """Parse the judge's JSON response into a JudgeResult."""
    try:
        # Strip markdown fences if the model added them anyway
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = "\n".join(cleaned.split("\n")[1:])
            cleaned = cleaned.rstrip("` \n")

        data = json.loads(cleaned)

        dimensions = {}
        weighted_score = 0.0
        for dim, weight in DIMENSION_WEIGHTS.items():
            if dim not in data:
                raise ValueError(f"Missing dimension: {dim}")
            score = float(data[dim]["score"])
            reason = str(data[dim].get("reason", ""))
            dimensions[dim] = DimensionScore(score=score, reason=reason)
            weighted_score += score * weight

        weighted_score = round(weighted_score, 2)
        return JudgeResult(
            weighted_score=weighted_score,
            passed=weighted_score >= MIN_PASS_SCORE,
            auto_approve_eligible=weighted_score >= AUTO_APPROVE_SCORE,
            dimensions=dimensions,
            judge_model=model,
            latency_ms=latency_ms,
            raw_response=raw,
        )

    except Exception as e:
        return _error_result(f"parse_error: {e} | raw: {raw[:200]}", model, latency_ms)


def _error_result(error_msg: str, model: str, latency_ms: int) -> JudgeResult:
    """Return a blocking result on any error (fail safe)."""
    dummy = DimensionScore(score=0.0, reason=error_msg)
    return JudgeResult(
        weighted_score=0.0,
        passed=False,
        auto_approve_eligible=False,
        dimensions={dim: dummy for dim in DIMENSION_WEIGHTS},
        judge_model=model,
        latency_ms=latency_ms,
        raw_response=error_msg,
    )


# ---------------------------------------------------------------------------
# Quality gate (rewrite loop) — mirrors the pipeline behaviour
# ---------------------------------------------------------------------------

def evaluate_with_retry(
    inp: JudgeInput,
    generate_fn,
    max_attempts: int = 2,
) -> tuple[str, JudgeResult, int]:
    """
    Generate a message and judge it. If it fails, regenerate once and
    judge again. Return (final_message, final_result, attempt_number).

    attempt_number meaning:
      0 = all attempts failed (caller should block this message)
      1 = passed on first generation
      2 = passed after one rewrite (judge_rewrite)

    Args:
        inp:          JudgeInput with prospect context
        generate_fn:  callable(inp) -> str — your message generation function
        max_attempts: how many generations to try (default 2)
    """
    for attempt in range(1, max_attempts + 1):
        message = generate_fn(inp)
        inp_with_msg = JudgeInput(
            message=message,
            company_name=inp.company_name,
            role=inp.role,
            industry=inp.industry,
            channel=inp.channel,
            campaign_tone=inp.campaign_tone,
            pain_points=inp.pain_points,
        )
        result = evaluate_message(inp_with_msg)

        if result.passed:
            return message, result, attempt

    # All attempts failed
    return message, result, 0


# ---------------------------------------------------------------------------
# Demo — runs if you execute this file directly
# ---------------------------------------------------------------------------

def _print_result(label: str, result: JudgeResult):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  Weighted score : {result.weighted_score:.1f} / 10")
    print(f"  Passed         : {result.passed}  (threshold: {MIN_PASS_SCORE})")
    print(f"  Auto-approve   : {result.auto_approve_eligible}  (threshold: {AUTO_APPROVE_SCORE})")
    print(f"  Latency        : {result.latency_ms} ms")
    print(f"\n  Dimension breakdown:")
    for dim, ds in result.dimensions.items():
        w = int(DIMENSION_WEIGHTS[dim] * 100)
        print(f"    {dim:<18} {ds.score:4.1f}/10  (weight {w}%)  — {ds.reason}")


if __name__ == "__main__":
    # ------------------------------------------------------------------
    # SAMPLE 1 — A well-personalized message (should score high)
    # ------------------------------------------------------------------
    good_message = """\
Subject: AI agents that handle your SaaS onboarding ops

Hi Marcus,

I noticed CloudScale just crossed 500 customers — congrats! At that stage
most SaaS founders tell us onboarding tickets eat 40% of CS bandwidth.

We build custom AI agents that handle the repetitive parts: account setup
emails, checklist nudges, integration troubleshooting FAQs — without adding
headcount.

Would a 15-minute call Thursday make sense to see if this fits CloudScale's
roadmap?

Best,
[Your Name]
[Your Company]"""

    inp_good = JudgeInput(
        message=good_message,
        company_name="CloudScale",
        role="Co-founder & CEO",
        industry="SaaS",
        channel="email",
        campaign_tone="professional",
        pain_points=["Scaling CS operations", "High onboarding ticket volume"],
    )

    # ------------------------------------------------------------------
    # SAMPLE 2 — Generic, no personalization (should score low)
    # ------------------------------------------------------------------
    bad_message = """\
Subject: Our AI solution can help your company

Hi,

We help companies with AI automation. Our product saves time and money.
Please let me know if you are interested in learning more.

Thanks"""

    inp_bad = JudgeInput(
        message=bad_message,
        company_name="CloudScale",
        role="Co-founder & CEO",
        industry="SaaS",
        channel="email",
        campaign_tone="professional",
        pain_points=["Scaling CS operations"],
    )

    if OPENROUTER_API_KEY == "YOUR_API_KEY_HERE":  # noqa: S105
        print("⚠️  Set OPENROUTER_API_KEY to run the live demo.")
        print("   Showing expected output structure instead:\n")
        print(json.dumps({
            "weighted_score": 7.85,
            "passed": True,
            "auto_approve_eligible": False,
            "dimensions": {
                "personalization": {"score": 8.5, "reason": "Mentions company milestone and specific pain point."},
                "cultural_fit":    {"score": 8.0, "reason": "Appropriate formality for target market."},
                "cta_strength":    {"score": 8.0, "reason": "Specific day and time suggested."},
                "tone_match":      {"score": 7.5, "reason": "Professional tone matches campaign setting."},
                "clarity":         {"score": 7.0, "reason": "Concise but could trim the middle paragraph."},
            }
        }, indent=2))
    else:
        result_good = evaluate_message(inp_good)
        _print_result("SAMPLE 1 — Personalized message", result_good)

        result_bad = evaluate_message(inp_bad)
        _print_result("SAMPLE 2 — Generic message", result_bad)
