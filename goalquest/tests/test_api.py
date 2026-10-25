from fastapi.testclient import TestClient
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.db import Base
from app.main import app, get_db, get_llm, get_graph
from app.graph import build_graph
from app.memory import MemoryStore
from tests.conftest import FakeLLM


@pytest.fixture
def client_and_db():
    eng = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(eng)
    TestingSessionLocal = sessionmaker(eng, expire_on_commit=False)
    fake_llm = FakeLLM()

    def override_get_db():
        with TestingSessionLocal() as db:
            yield db

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_llm] = lambda: fake_llm
    app.dependency_overrides[get_graph] = lambda: build_graph(TestingSessionLocal, fake_llm, seed=0)

    client = TestClient(app)
    yield client, TestingSessionLocal
    app.dependency_overrides.clear()


def test_users_and_goals_flow(client_and_db):
    client, _ = client_and_db

    # Test root dashboard UI route & favicon
    ui_res = client.get("/")
    assert ui_res.status_code == 200
    assert "GoalQuest" in ui_res.text

    fav_res = client.get("/favicon.ico")
    assert fav_res.status_code == 200
    assert "svg" in fav_res.headers.get("content-type", "")

    # Create user
    res = client.post("/users", json={"name": "Alice", "coach_style": "mentor"})
    assert res.status_code == 200
    uid = res.json()["user_id"]

    # Goal for non-existent user -> 404
    bad_res = client.post("/goals", json={
        "user_id": 9999,
        "goal": "Master Rust",
        "why": "Systems programming",
        "deadline_days": 30,
        "weekly_hours": 6.0,
    })
    assert bad_res.status_code == 404

    # Goal for real user -> 200
    goal_res = client.post("/goals", json={
        "user_id": uid,
        "goal": "Master Rust",
        "why": "Systems programming",
        "deadline_days": 30,
        "weekly_hours": 6.0,
    })
    assert goal_res.status_code == 200
    gid = goal_res.json()["goal_id"]

    # Dashboard
    dash = client.get(f"/dashboard/{uid}/{gid}")
    assert dash.status_code == 200
    assert dash.json()["goal"] == "Learn linear algebra"  # from FakeLLM plan

    # Goal tasks endpoint
    tasks_res = client.get(f"/goals/{gid}/tasks")
    assert tasks_res.status_code == 200
    assert len(tasks_res.json()) >= 1
    assert "xp" in tasks_res.json()[0]

    # Dashboard 404s
    assert client.get(f"/dashboard/9999/{gid}").status_code == 404
    assert client.get(f"/dashboard/{uid}/9999").status_code == 404


def test_checkin_and_chat(client_and_db):
    client, _ = client_and_db
    uid = client.post("/users", json={"name": "Bob"}).json()["user_id"]
    gid = client.post("/goals", json={
        "user_id": uid, "goal": "Algebra", "why": "GATE", "deadline_days": 28, "weekly_hours": 5.0,
    }).json()["goal_id"]

    # Checkin with non-existent goal -> 404
    assert client.post("/checkin", json={
        "user_id": uid, "goal_id": 9999,
        "checkin": {"energy": 3, "mood": 4, "minutes_focused": 45, "distractions": 1, "note": "good session"}
    }).status_code == 404

    # Valid checkin
    res = client.post("/checkin", json={
        "user_id": uid, "goal_id": gid,
        "checkin": {"energy": 4, "mood": 4, "minutes_focused": 45, "distractions": 1, "note": "good session"}
    })
    assert res.status_code == 200
    data = res.json()
    assert "band" in data and "quests" in data and "message" in data

    # Chat
    chat_res = client.post("/chat", json={"user_id": uid, "goal_id": gid, "message": "How am I doing?"})
    assert chat_res.status_code == 200
    assert "reply" in chat_res.json()

    # Chat 404
    assert client.post("/chat", json={"user_id": 9999, "goal_id": gid, "message": "hi"}).status_code == 404


def test_quiz_and_verify_flow(client_and_db):
    client, db_session = client_and_db
    uid = client.post("/users", json={"name": "Charlie"}).json()["user_id"]
    gid = client.post("/goals", json={
        "user_id": uid, "goal": "Linear Algebra", "why": "Exam", "deadline_days": 28, "weekly_hours": 5.0,
    }).json()["goal_id"]

    # Verify before requesting quiz -> 409
    assert client.post("/tasks/1/verify", json={"user_id": uid, "answer": "basis"}).status_code == 409

    # Request quiz on non-existent task -> 404
    assert client.post("/tasks/9999/quiz").status_code == 404

    # Request quiz on valid task 1
    q_res = client.post("/tasks/1/quiz")
    assert q_res.status_code == 200
    assert "questions" in q_res.json()

    # Verify quiz on valid task 1
    v_res = client.post("/tasks/1/verify", json={"user_id": uid, "answer": "A basis spans the vector space"})
    assert v_res.status_code == 200
    assert v_res.json()["verification"]["passed"] is True
    assert "reward" in v_res.json()
    assert v_res.json()["reward"]["xp_awarded"] > 0


def test_memory_store_fallback():
    mem = MemoryStore(path="./.non_existent_chroma_path")
    mem.add(42, "Remember to focus on fundamentals and basis vectors", "reflection", "doc1")
    matches = mem.search(42, "basis fundamentals")
    assert len(matches) >= 1
    assert "fundamentals" in matches[0]

