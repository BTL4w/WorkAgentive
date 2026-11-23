"""Identify the literal topic authorized by a single public search request."""

import re

from work_management_ai.agents.orchestrator.contracts import MessageSpan

_PUBLIC_TOPIC_PATTERNS = (
    re.compile(
        r"(?:please )?(?:search|check) (?:the )?(?:public )?(?:web|internet) for (?P<topic>.+)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:please )?(?:find|look up|search for) (?P<topic>.+?) "
        r"(?:on|from) (?:the )?(?:public )?(?:web|internet)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:tìm(?: kiếm)?|tra cứu)(?: giúp tôi)? (?P<topic>.+?) "
        r"(?:trên|từ) (?:web(?: công khai)?|internet|nguồn (?:web )?công khai)",
        re.IGNORECASE,
    ),
)
_COMPOUND_TOPIC = re.compile(
    r"[,;]|:(?!//)|\b(?:but|however|except|keep|do not|don't|without sharing|"
    r"nhưng|tuy nhiên|ngoại trừ|trừ khi|giữ kín|đừng|không (?:gửi|chia sẻ|tiết lộ))\b",
    re.IGNORECASE,
)


def select_public_topic(original: str, objective: str = "") -> MessageSpan | None:
    """Narrow the affirmative topic only when the objective is a literal excerpt of it."""
    message = original.strip().rstrip(".!?")
    if re.search(r"[\r\n]|[.!?]\s+\S", message):
        return None
    for pattern in _PUBLIC_TOPIC_PATTERNS:
        match = pattern.fullmatch(message)
        if match is None:
            continue
        raw_topic = match["topic"]
        topic = raw_topic.strip()
        if not 1 <= len(topic) <= 500 or _COMPOUND_TOPIC.search(topic):
            return None
        topic_start = (
            original.index(message)
            + match.start("topic")
            + len(raw_topic)
            - len(raw_topic.lstrip())
        )
        query = objective.strip()
        offset = topic.find(query) if query else -1
        if offset < 0:
            query, offset = topic, 0
        start = topic_start + offset
        return MessageSpan(start=start, end=start + len(query), text=query)
    return None
