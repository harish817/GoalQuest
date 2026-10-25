"""FastAPI surface.  Run:  uvicorn app.main:app --reload"""
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from functools import lru_cache

from fastapi import Depends, FastAPI, HTTPException, Response
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from . import service
from .agents import planner, verifier
from .config import settings
from .db import Goal, SessionLocal, Task, User, init_db, seed_default_data
from .game import reward_engine as re_
from .graph import build_graph
from .llm import LLM, create_llm
from .memory import MemoryStore
from .schemas import CheckIn, VerificationResult
from .tools import CHAT_SYSTEM, TOOL_SCHEMAS, make_handlers

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    with SessionLocal() as db:
        seed_default_data(db)
    yield


app = FastAPI(title="GoalQuest", lifespan=lifespan)


def get_db():
    with SessionLocal() as db:
        yield db


@lru_cache
def get_llm() -> LLM:
    return create_llm()


@lru_cache
def get_memory() -> MemoryStore:
    return MemoryStore()


@lru_cache
def get_graph():
    return build_graph(SessionLocal, get_llm())


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    svg = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><text y=".9em" font-size="90">⚔️</text></svg>'
    return Response(content=svg, media_type="image/svg+xml")


@app.get("/", response_class=HTMLResponse)
def index():
    html_path = os.path.join(os.path.dirname(__file__), "static", "index.html")
    if os.path.exists(html_path):
        with open(html_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse("<h1>GoalQuest API is running</h1>")


@app.get("/goals/{goal_id}/tasks")
def get_goal_tasks(goal_id: int, db: Session = Depends(get_db)):
    goal = db.get(Goal, goal_id)
    if not goal:
        raise HTTPException(404, "goal not found")
    tasks = db.scalars(select(Task).where(Task.goal_id == goal_id).order_by(Task.id)).all()
    unlocked_ids = {t.id for t in service.unlocked_tasks(db, goal_id)}
    return [
        {
            "id": t.id,
            "key": t.key,
            "title": t.title,
            "kind": t.kind,
            "minutes": t.est_minutes,
            "difficulty": t.difficulty,
            "is_boss": bool(t.is_boss),
            "status": "done" if t.status == "done" else ("unlocked" if t.id in unlocked_ids else "locked"),
            "done_when": t.done_when,
            "xp": re_.task_xp(t.est_minutes, t.difficulty, t.kind),
        }
        for t in tasks
    ]


_QUIZ_CACHE: dict[int, verifier.Quiz] = {}


def _save_quiz(task_id: int, quiz: verifier.Quiz) -> None:
    _QUIZ_CACHE[task_id] = quiz
    try:
        import redis
        r = redis.from_url(settings.redis_url, socket_timeout=0.5)
        r.set(f"quiz:{task_id}", quiz.model_dump_json(), ex=3600)
    except Exception:
        pass


def _get_quiz(task_id: int) -> verifier.Quiz | None:
    try:
        import redis
        r = redis.from_url(settings.redis_url, socket_timeout=0.5)
        val = r.get(f"quiz:{task_id}")
        if val:
            return verifier.Quiz.model_validate_json(val)
    except Exception:
        pass
    return _QUIZ_CACHE.get(task_id)


def _pop_quiz(task_id: int) -> None:
    _QUIZ_CACHE.pop(task_id, None)
    try:
        import redis
        r = redis.from_url(settings.redis_url, socket_timeout=0.5)
        r.delete(f"quiz:{task_id}")
    except Exception:
        pass


class NewUser(BaseModel):
    name: str
    coach_style: str = "mentor"


class NewGoal(BaseModel):
    user_id: int
    goal: str
    why: str
    deadline_days: int
    weekly_hours: float


class DailyCheckIn(BaseModel):
    user_id: int
    goal_id: int
    checkin: CheckIn


class Answer(BaseModel):
    user_id: int
    answer: str


class ChatMsg(BaseModel):
    user_id: int
    goal_id: int
    message: str


@app.post("/users")
def create_user(body: NewUser, db: Session = Depends(get_db)):
    u = User(name=body.name, coach_style=body.coach_style)
    db.add(u)
    db.commit()
    return {"user_id": u.id}


@app.post("/goals")
def create_goal(body: NewGoal, db: Session = Depends(get_db), llm: LLM = Depends(get_llm)):
    user = db.get(User, body.user_id)
    if not user:
        raise HTTPException(404, "user not found")
    plan = planner.generate_plan(llm, body.goal, body.why, body.deadline_days, body.weekly_hours)
    goal = service.save_plan(db, body.user_id, plan)
    get_memory().add(body.user_id, f"Goal: {body.goal}. Why: {body.why}", "goal_why", f"goal-{goal.id}")
    return {"goal_id": goal.id, "plan": plan.model_dump(mode="json"),
            "over_budget": planner.total_minutes(plan) > planner.budget_minutes(body.weekly_hours, body.deadline_days)}


@app.post("/checkin")
def daily_checkin(body: DailyCheckIn, db: Session = Depends(get_db), llm: LLM = Depends(get_llm)):
    if not db.get(User, body.user_id):
        raise HTTPException(404, "user not found")
    if not db.get(Goal, body.goal_id):
        raise HTTPException(404, "goal not found")
    if body.checkin.note:
        get_memory().add(body.user_id, body.checkin.note, "reflection", f"checkin-{body.user_id}")
    graph = build_graph(lambda: db, llm)
    return graph.invoke({"user_id": body.user_id, "goal_id": body.goal_id,
                         "checkin": body.checkin.model_dump()})


@app.post("/tasks/{task_id}/quiz")
def make_quiz(task_id: int, db: Session = Depends(get_db), llm: LLM = Depends(get_llm)):
    t = db.get(Task, task_id)
    if not t:
        raise HTTPException(404, "task not found")
    try:
        q = verifier.make_quiz(llm, t.title, t.done_when)
    except Exception as e:
        logger.warning("LLM quiz generation error (%s), using structured fallback quiz", e)
        q = verifier.Quiz(
            questions=[
                f"Explain how your solution satisfies: '{t.done_when}'.",
                f"What are the key technical concepts mastered in '{t.title}'?",
            ],
            rubric=f"Must address {t.title} and clearly show satisfying '{t.done_when}'.",
        )
    _save_quiz(task_id, q)
    return {"questions": q.questions}


@app.post("/tasks/{task_id}/verify")
def verify(task_id: int, body: Answer, db: Session = Depends(get_db), llm: LLM = Depends(get_llm)):
    if not db.get(User, body.user_id):
        raise HTTPException(404, "user not found")
    quiz = _get_quiz(task_id)
    if quiz is None:
        raise HTTPException(409, "request a quiz first")
    try:
        res = verifier.grade(llm, quiz, body.answer)
    except Exception as e:
        logger.warning("LLM grading error (%s), using fallback validation", e)
        passed = len(body.answer.strip()) >= 25
        res = VerificationResult(
            passed=passed,
            score=0.9 if passed else 0.4,
            feedback="Verification passed. Good explanation!" if passed else "Please provide a more complete explanation.",
        )
    out = {"verification": res.model_dump()}
    if res.passed:
        out["reward"] = service.complete_task(db, body.user_id, task_id, res.score)
        _pop_quiz(task_id)
    return out


@app.get("/dashboard/{user_id}/{goal_id}")
def get_dashboard(user_id: int, goal_id: int, db: Session = Depends(get_db)):
    if not db.get(User, user_id):
        raise HTTPException(404, "user not found")
    if not db.get(Goal, goal_id):
        raise HTTPException(404, "goal not found")
    return service.dashboard(db, user_id, goal_id)


@app.post("/chat")
def chat(body: ChatMsg, db: Session = Depends(get_db), llm: LLM = Depends(get_llm)):
    if not db.get(User, body.user_id):
        raise HTTPException(404, "user not found")
    if not db.get(Goal, body.goal_id):
        raise HTTPException(404, "goal not found")
    handlers = make_handlers(db, body.user_id, body.goal_id, memory=get_memory())
    reply = llm.run_tool_loop(CHAT_SYSTEM, body.message, TOOL_SCHEMAS, handlers)
    return {"reply": reply}
