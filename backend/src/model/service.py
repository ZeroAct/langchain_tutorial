from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseLLM
from langchain_core.messages import BaseMessage
from langchain_ollama import OllamaEmbeddings, OllamaLLM

from .exceptions import *


class ModelService:
    try:
        models: dict[str, tuple[BaseLLM, Embeddings]] = {
            "llama3.2": (OllamaLLM(model="llama3.2"), OllamaEmbeddings(model="llama3.2"))
        }
    except Exception as e:
        raise ModelInitializationError("Failed to initialize models") from e

    async def get_model_list(self) -> list[str]:
        return list(self.models.keys())

    async def invoke(self, model: str, messages: list[BaseMessage]) -> str:
        if model not in self.models:
            raise ModelNotFoundError(f"Model {model} not found") from e

        model: BaseLLM = self.models[model][0]
        try:
            result = await model.ainvoke(messages)
            return result
        except Exception as e:
            raise ModelInvokeError("Failed to invoke model") from e

    async def embed(self, model: str, content: str) -> list[float]:
        if model not in self.models:
            raise ValueError(f"Model {model} not found")

        model: Embeddings = self.models[model][1]
        try:
            result = await model.aembed_query(content)
            return result
        except Exception as e:
            raise ModelEmbedError("Failed to embed content") from e


model_service = ModelService()
