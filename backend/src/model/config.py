from pydantic_settings import BaseSettings


class ModelConfig(BaseSettings):
    ollama_cfg: dict = {
        "host": "localhost:11434",
        "model_list": ["llama3.2"],
    }
