from pydantic import BaseModel


class AppSettings(BaseModel):
    name: str
    version: str
    environment: str


class LLMSettings(BaseModel):
    provider: str
    model: str


class LoggingSettings(BaseModel):
    level: str


class MemorySettings(BaseModel):
    enabled: bool


class Settings(BaseModel):
    app: AppSettings
    llm: LLMSettings
    logging: LoggingSettings
    memory: MemorySettings
