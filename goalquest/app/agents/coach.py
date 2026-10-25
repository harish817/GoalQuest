from __future__ import annotations

from ..llm import LLM

STYLE_PROMPTS = {
    "celebrate": "Celebrate the specific win with genuine energy, then point to the next step.",
    "gentle_nudge": "Be warm and low-pressure. Acknowledge how hard it can be. Suggest one small action.",
    "challenge": "Offer a friendly stretch challenge with a clear target and XP bonus.",
    "why_reminder": "Reconnect to the user's own stated reason for the goal, quoting it naturally.",
    "tiny_step": "Shrink the task: propose a 5-minute version to just get started.",
}
PERSONAS = {
    "mentor": "calm, wise and encouraging",
    "sergeant": "direct, high-energy, no-nonsense but never insulting",
    "sidekick": "playful, funny and upbeat",
}

COACH_SYSTEM = """You are the user's motivational coach inside a gamified goal app.
Persona: {persona}. Message style for this turn: {style}
Hard rules: max 60 words; never shame or guilt; no medical advice; reference real data you were given
(streak, XP, progress, today's quests); do not invent numbers. If risk is high, prioritise kindness and a tiny step."""


def write_message(llm: LLM, style: str, persona: str, facts: dict) -> str:
    system = COACH_SYSTEM.format(persona=PERSONAS.get(persona, PERSONAS["mentor"]),
                                 style=STYLE_PROMPTS[style])
    return llm.text(system, f"Facts: {facts}", max_tokens=200)


FOCUS_TIPS = {
    "distracted": ["Phone in another room before the block starts.",
                   "Keep a 'parking lot' note for stray ideas; capture, don't chase.",
                   "Try a 25/5 block; shorter blocks are easier to protect."],
    "low_energy": ["Do the easiest task first for a quick win.",
                   "Take a 10-minute walk, then do one 25-minute block.",
                   "Schedule demanding work at your best hours (see your focus log)."],
    "default": ["Write the ONE thing you'll finish this block before you start.",
                "Close every tab unrelated to the task.",
                "Silence notifications for the length of the block."],
}


def focus_tips(energy: int, distractions: int) -> list[str]:
    if distractions >= 3:
        return FOCUS_TIPS["distracted"]
    if energy <= 2:
        return FOCUS_TIPS["low_energy"]
    return FOCUS_TIPS["default"]
