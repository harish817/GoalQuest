# ⚔️ GoalQuest: Gamified Goal-Achievement Engine

GoalQuest is an agentic, gamified goal-achievement engine and RPG dashboard. It transforms complex, long-term learning goals into structured Chapter $\rightarrow$ Quest $\rightarrow$ Task directed acyclic graphs (DAGs), enforcing verified task completion, tamper-evident rewards, adaptive stamina matching, and AI coaching.

---

## 🌟 Key Features

* **🎮 Deterministic RPG Game Mechanics:**
  * **Levels & Ranks:** Progress through *Apprentice*, *Practitioner*, *Adept*, *Expert*, and *Master*.
  * **Tamper-Evident XP Ledger:** Append-only database ledger with diminishing returns past 60% of the daily cap (400 XP).
  * **Streaks & Freeze Shields:** Earn freeze tokens every 7 active days. Multiplier adds +2% per streak day up to 1.5x.
  * **Boss Challenges:** Integrative capstone quests in each chapter with a 1.5x multiplier and the *Boss Slayer* badge.
  * **Variable-Ratio Loot Drops:** 30% drop rate for bonus XP, streak shields, and tokens on verified completions.
  * **Achievement Badges:** *First Blood*, *Deep Work*, *Comeback Kid*, *Boss Slayer*, *Week Warrior*, and *Month Monk*.

* **🧠 Machine Learning & Cognitive Models:**
  * **Thompson-Sampling Multi-Armed Bandit:** Learns which coaching persona and tone (*celebrate*, *challenge*, *gentle nudge*, *why reminder*, *tiny step*) maximizes 24-hour task completion.
  * **Dropout Risk Model:** Predicts student churn from 7-day rolling check-in features (focus slope, distraction rate, completion rate) and triggers an automatic recovery on-ramp.
  * **Bayesian Knowledge Tracing (BKT) & SM-2:** Models learner skill acquisition and spaced repetition intervals for reviews.
  * **Duration Calibrator:** Corrects optimistic task estimates using an exponentially weighted moving log-ratio.

* **🤖 Dual LLM Support (Groq & Anthropic):**
  * Built-in support for Groq (e.g., `openai/gpt-oss-120b`, `qwen/qwen3.8-27b`) via native function calling with automatic model fallbacks.
  * Support for Anthropic Claude (`claude-sonnet-5-5`).
  * Fail-safe verifier and quiz evaluation fallbacks.

* **🔌 Model Context Protocol (MCP) Server:**
  * Built-in MCP server (`goalquest/app/mcp_server.py`) allowing Claude Desktop and Claude Code to read progress, view active quests, and log check-ins.

* **🎨 Interactive RPG Web UI:**
  * Dark RPG arcade dashboard with live character HUD, stamina slider, quiz verification battle modal with synthesized Web Audio chimes, canvas confetti, and AI coach chat drawer.

---

## 🚀 Quick Start

### 1. Prerequisites
* Python 3.11+
* [uv](https://docs.astral.sh/uv/) (recommended) or `pip`

### 2. Installation
Clone the repository and install dependencies:

```bash
git clone https://github.com/harish817/goalquest.git
cd goalquest
uv sync
```

Or using pip:
```bash
pip install -r goalquest/requirements.txt
```

### 3. Environment Setup
Copy the example environment file and add your API key:

```bash
cp .env.example .env
```

Edit `.env` to include your credentials:
```env
GROQ_API_KEY=gsk_your_groq_api_key
GQ_MODEL=openai/gpt-oss-120b
GQ_DATABASE_URL=sqlite:///./goalquest.db
```

### 4. Run the Application
Launch the server:
```bash
python main.py
```
Open **[http://127.0.0.1:8000](http://127.0.0.1:8000)** in your browser to view the live RPG dashboard.

---

## 🧪 Testing

Run the full test suite (27 unit & integration tests):

```bash
pytest
```

---

## 📡 API Overview

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Serves the interactive RPG Web UI |
| `POST` | `/users` | Create a new user profile |
| `POST` | `/goals` | AI planner decomposes goal into Chapter/Quest DAG |
| `GET` | `/goals/{goal_id}/tasks` | Retrieve all quests and dependencies for a goal |
| `POST` | `/checkin` | Daily check-in, dropout risk assessment, and bandit coach loop |
| `POST` | `/tasks/{id}/quiz` | Generate verification quiz questions for a task |
| `POST` | `/tasks/{id}/verify` | Grade task completion, award XP, update streak, roll loot |
| `GET` | `/dashboard/{uid}/{gid}` | Retrieve player HUD, progress, streak, badges, and pace |
| `POST` | `/chat` | Conversational goal coach with tool-calling capabilities |

---

## 🛠️ MCP Server Integration

To use GoalQuest directly inside Claude Desktop, add this to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "goalquest": {
      "command": "python",
      "args": ["-m", "app.mcp_server"],
      "cwd": "/path/to/goalquest/goalquest"
    }
  }
}
```

---

## 📜 License
MIT

