from __future__ import annotations

from pydantic import BaseModel, Field

from ..llm import LLM
from ..ml.knowledge import BKT
from ..schemas import VerificationResult


class Quiz(BaseModel):
    questions: list[str] = Field(min_length=1, max_length=3)
    rubric: str


QUIZ_SYSTEM = ("You verify that a learner really completed a task. Write 1-3 short questions "
               "(or one teach-back prompt) that can only be answered well by someone who did the work. "
               "Also write a concise grading rubric.")
GRADE_SYSTEM = ("You grade a learner's answer against a rubric. Be fair but do not pass vague or "
                "copy-pasted answers. score is 0-1; passed means score >= 0.6. Feedback must be specific and kind.")


def make_quiz(llm: LLM, title: str, done_when: str) -> Quiz:
    return llm.structured(QUIZ_SYSTEM, f"Task: {title}\nDefinition of done: {done_when}", Quiz)


def grade(llm: LLM, quiz: Quiz, answer: str, skill_state: BKT | None = None) -> VerificationResult:
    res = llm.structured(
        GRADE_SYSTEM,
        f"Questions: {quiz.questions}\nRubric: {quiz.rubric}\nLearner answer: {answer}",
        VerificationResult)
    if skill_state is not None:
        skill_state.update(res.passed)
    return res
