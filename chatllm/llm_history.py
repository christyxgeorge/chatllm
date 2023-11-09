from pydantic import BaseModel


class LLMHistoryItem(BaseModel):
    text: str
    role: str
    token_count: int


class LLMHistory:
    entries: list[LLMHistoryItem] = []

    def add(self, text: str, role: str, token_count: int) -> None:
        self.entries.append(LLMHistoryItem(text=text, role=role, token_count=token_count))
