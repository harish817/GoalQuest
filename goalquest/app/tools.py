"""Tools exposed to Claude for the free-form 'chat with your agent' mode.

The model can look things up and take actions, but every state change still goes
through service.py. Importantly, `complete_task` requires a verification score,
so the model cannot award XP by simply asserting that work was done.
"""
from __future__ import annotations

from typing import Callable

from sqlalchemy.orm import Session

from . import service
from .agents import coach
from .memory import MemoryStore

TOOL_SCHEMAS = [
    {"name": "get_dashboard", "description": "Progress, level, XP, streak, badges and pace for the user's goal.",
     "input_schema": {"type": "object", "properties": {}, "required": []}},
    {"name": "get_todays_quests",
     "description": "Choose today's quests, matched to the user's current energy (1-5).",
     "input_schema": {"type": "object", "properties": {"energy": {"type": "integer", "minimum": 1, "maximum": 5}},
                      "required": ["energy"]}},
    {"name": "complete_task",
     "description": "Mark a task complete and award XP. ONLY call after the verifier produced a score; "
                    "pass that exact score. Never guess the score.",
     "input_schema": {"type": "object", "properties": {
         "task_id": {"type": "integer"},
         "verified_score": {"type": "number", "minimum": 0, "maximum": 1}},
         "required": ["task_id", "verified_score"]}},
    {"name": "get_focus_tips", "description": "Concentration tips tailored to energy and distraction count.",
     "input_schema": {"type": "object", "properties": {
         "energy": {"type": "integer"}, "distractions": {"type": "integer"}},
         "required": ["energy", "distractions"]}},
    {"name": "recall_user_context", "description": "Semantic search over the user's past reflections and motivators.",
     "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}},
]


def make_handlers(db: Session, user_id: int, goal_id: int, memory: MemoryStore | None = None) -> dict[str, Callable]:
    def get_dashboard() -> dict:
        return service.dashboard(db, user_id, goal_id)

    def get_todays_quests(energy: int) -> list[dict]:
        return [{"id": t.id, "title": t.title, "minutes": t.est_minutes, "difficulty": t.difficulty}
                for t in service.pick_today(db, goal_id, energy)]

    def complete_task(task_id: int, verified_score: float) -> dict:
        return service.complete_task(db, user_id, task_id, verified_score)

    def get_focus_tips(energy: int, distractions: int) -> list[str]:
        return coach.focus_tips(energy, distractions)

    def recall_user_context(query: str) -> list[str]:
        return memory.search(user_id, query) if memory else []

    return {"get_dashboard": get_dashboard, "get_todays_quests": get_todays_quests,
            "complete_task": complete_task, "get_focus_tips": get_focus_tips,
            "recall_user_context": recall_user_context}


CHAT_SYSTEM = """You are GoalQuest, a warm but honest goal coach with game mechanics.
Use tools for facts (never invent XP, streaks or progress). To award XP for a task, the user must
first demonstrate completion; only pass a verified_score you were given. Keep replies under 80 words."""
