"""GoalQuest as an MCP server, so any MCP client (Claude Desktop, Claude Code, ...)
can read your progress and log work. Run:  python -m app.mcp_server

Claude Desktop config (claude_desktop_config.json):
  {"mcpServers": {"goalquest": {"command": "python", "args": ["-m", "app.mcp_server"],
                                "cwd": "/path/to/goalquest"}}}
"""
from __future__ import annotations

try:  # mcp SDK >= 2.0 renamed FastMCP -> MCPServer (same decorator API)
    from mcp.server.mcpserver import MCPServer as _Server
except ImportError:  # mcp SDK 1.x
    from mcp.server.fastmcp import FastMCP as _Server

from . import service
from .db import Goal, SessionLocal, User, init_db
from .schemas import CheckIn

mcp = _Server("goalquest")
init_db()

# For a single-user local setup. For multi-user, resolve the user from auth instead.
USER_ID, GOAL_ID = 1, 1


@mcp.tool()
def goalquest_dashboard() -> dict:
    """Current level, XP, streak, badges and days ahead/behind pace."""
    with SessionLocal() as db:
        if not db.get(User, USER_ID) or not db.get(Goal, GOAL_ID):
            return {"error": f"User {USER_ID} or Goal {GOAL_ID} not found. Please create them first."}
        return service.dashboard(db, USER_ID, GOAL_ID)


@mcp.tool()
def goalquest_todays_quests(energy: int = 3) -> list[dict]:
    """Today's recommended quests for an energy level 1-5."""
    with SessionLocal() as db:
        if not db.get(Goal, GOAL_ID):
            return []
        return [{"id": t.id, "title": t.title, "minutes": t.est_minutes}
                for t in service.pick_today(db, GOAL_ID, energy)]


@mcp.tool()
def goalquest_log_checkin(energy: int, mood: int, minutes_focused: int = 0, distractions: int = 0) -> str:
    """Log today's energy/mood (1-5) and focus stats."""
    with SessionLocal() as db:
        if not db.get(User, USER_ID):
            return f"error: user {USER_ID} not found"
        service.record_checkin(db, USER_ID, CheckIn(energy=energy, mood=mood,
                                                    minutes_focused=minutes_focused, distractions=distractions))
    return "logged"


@mcp.resource("goalquest://plan")
def plan() -> dict:
    """The full goal plan (chapters, quests, tasks)."""
    with SessionLocal() as db:
        g = db.get(Goal, GOAL_ID)
        return g.plan_json if g else {"error": f"Goal {GOAL_ID} not found"}


if __name__ == "__main__":
    mcp.run()
