import uvicorn
from app.main import app


def main():
    print("Starting GoalQuest FastAPI server on http://127.0.0.1:8000...")
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)


if __name__ == "__main__":
    main()
