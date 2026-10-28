"""Thin public adapter shared by API callers."""

from functools import lru_cache
from agent.session import ChatSession


@lru_cache(maxsize=1)
def get_session():
    return ChatSession.build()


def answer_question(question):
    return get_session().ask(question)
