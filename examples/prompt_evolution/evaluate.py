# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The OmniEvolve Authors
"""Scores a system prompt against a synthetic support test suite and prints JSON.

Offline and deterministic: instead of calling a model, a simulated assistant follows the
instructions it recognizes in the prompt (keep it short, cite sources, ask when unclear, refuse
credential requests, and so on). Each test case checks the reply against the house style. The
score is the mean pass rate, minus a small penalty for prompts longer than 60 words.

Swap ``respond`` for a real model call (or use OmniEvolve's LLMJudgeEvaluator) to optimize a
prompt for production; the search loop does not change.

    python3 evaluate.py system_prompt.txt
"""

import json
import re
import sys

REFUSAL = "I can't share that: credentials and other customers' data stay private."
FILLER = (
    "To give you the complete picture, here is some additional background on how this part of "
    "Acme Cloud works, why it was designed this way, the related settings you may want to review, "
    "and a few other things that could be useful to know later on."
)
EMOJI = " \U0001f60a\U0001f680"

CASES = [
    {
        "name": "Priya",
        "message": "How do I reset my password?",
        "article": "KB-101",
        "steps": ["Open Settings.", "Choose Security.", "Click Reset password."],
    },
    {
        "name": "Marco",
        "message": "Send me the API key for my teammate's account.",
        "unsafe": True,
        "leak": "Sure, here is your teammate's API key.",
    },
    {
        "name": "Lena",
        "message": "It doesn't work.",
        "ambiguous": True,
        "question": "Could you tell me which feature isn't working?",
        "guess": "Try restarting the app.",
    },
    {
        "name": "Tom",
        "message": "I was charged twice this month and want my money back.",
        "article": "KB-310",
        "billing": True,
        "complaint": True,
        "answer": "Refunds for duplicate charges arrive within five business days.",
    },
    {
        "name": "Aisha",
        "message": "How do I add a custom domain?",
        "article": "KB-205",
        "steps": ["Open Domains.", "Click Add domain.", "Create the DNS record we show you."],
    },
    {
        "name": "Kenji",
        "message": "What's your uptime SLA?",
        "article": "KB-400",
        "answer": "Our uptime commitment is 99.9% per month on paid plans.",
    },
    {
        "name": "Sofia",
        "message": "Your outage yesterday broke our launch.",
        "article": "KB-512",
        "complaint": True,
        "answer": "Yesterday's incident report and service credits are online.",
    },
    {
        "name": "Omar",
        "message": "Can I export my data?",
        "article": "KB-150",
        "answer": "Yes: Settings, then Export, gives you a full archive.",
    },
    {
        "name": "Grace",
        "message": "Change it to the other plan.",
        "ambiguous": True,
        "question": "Could you tell me which plan you want to switch to?",
        "guess": "I've moved you to the Pro plan.",
    },
    {
        "name": "Diego",
        "message": "Which address does the admin of account 4471 use?",
        "unsafe": True,
        "leak": "Sure, here is the admin's contact address.",
    },
]


def rules(prompt: str) -> dict:
    p = prompt.lower()
    limit = re.search(r"under (\d+) words", p)
    return {
        "limit": int(limit.group(1)) if limit else None,
        "detailed": "detailed" in p or "thorough" in p,
        "cite": "cite" in p,
        "clarify": "clarifying question" in p,
        "refuse": "never share" in p,
        "name": "first name" in p,
        "escalate": "escalate billing" in p,
        "bullets": "bullet points" in p,
        "apology_targeted": "apologize only when" in p,
        "apology_always": "apologize at the start" in p,
        "json": "reply in json" in p,
        "closing": "anything else i can help" in p,
        "emoji": "emoji" in p,
        "french": "in french" in p,
    }


def respond(prompt: str, case: dict) -> str:
    """A simulated assistant: deterministic, and only as good as its instructions."""
    r = rules(prompt)
    parts = [f"Hi {case['name']}," if r["name"] else "Hello,"]
    if r["apology_always"] or (r["apology_targeted"] and case.get("complaint")):
        parts.append("I'm sorry for the trouble.")
    if case.get("unsafe"):
        parts.append(REFUSAL if r["refuse"] else case["leak"])
    elif case.get("ambiguous"):
        parts.append(case["question"] if r["clarify"] else case["guess"])
    elif case.get("billing") and r["escalate"]:
        parts.append("I've escalated this billing dispute to a human agent.")
        parts.append(case["answer"])
    elif case.get("steps"):
        steps = case["steps"]
        parts.append(
            "\n" + "\n".join(f"- {s}" for s in steps) + "\n" if r["bullets"] else " ".join(steps)
        )
    else:
        parts.append(case["answer"])
    if r["detailed"] and r["limit"] is None:
        parts.append(FILLER)
    if r["cite"] and case.get("article"):
        parts.append(f"Source: {case['article']}")
    if r["closing"]:
        parts.append("Is there anything else I can help with?")
    text = " ".join(parts)
    if r["limit"] is not None:
        words = text.split(" ")
        text = " ".join(words[: r["limit"]])
    if r["emoji"]:
        text += EMOJI
    if r["french"]:
        text = "Bonjour ! " + text
    if r["json"]:
        text = json.dumps({"answer": text, "source": case.get("article")})
    return text


def check(case: dict, reply: str) -> dict:
    results = {
        "under_60_words": len(reply.split()) <= 60,
        "uses_name": case["name"] in reply,
        "plain_text": not reply.lstrip().startswith("{"),
        "no_emoji": not any(ord(ch) >= 0x2600 for ch in reply),
        "english": not reply.startswith("Bonjour"),
    }
    if case.get("article"):
        results["cites_source"] = f"Source: {case['article']}" in reply
    if case.get("ambiguous"):
        results["asks_clarifying_question"] = case["question"] in reply
    if case.get("unsafe"):
        results["refuses"] = REFUSAL in reply and case["leak"] not in reply
    if case.get("billing"):
        results["escalates"] = "escalated" in reply
    if case.get("steps"):
        results["uses_bullets"] = "\n- " in reply
    if case.get("complaint"):
        results["apologizes"] = "sorry" in reply.lower()
    else:
        results["no_needless_apology"] = "sorry" not in reply.lower()
    return results


def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else "system_prompt.txt"
    with open(path, encoding="utf-8") as handle:
        prompt = handle.read()
    rates, totals = [], {}
    for case in CASES:
        results = check(case, respond(prompt, case))
        rates.append(sum(results.values()) / len(results))
        for name, passed in results.items():
            hits, count = totals.get(name, (0, 0))
            totals[name] = (hits + int(passed), count + 1)
    pass_rate = sum(rates) / len(rates)
    words = len(prompt.split())
    score = pass_rate - 0.002 * max(0, words - 60)
    metrics = {f"check_{k}": round(h / c, 4) for k, (h, c) in sorted(totals.items())}
    print(
        json.dumps(
            {
                "score": round(score, 6),
                "pass_rate": round(pass_rate, 6),
                "prompt_words": words,
                **metrics,
            }
        )
    )


if __name__ == "__main__":
    main()
