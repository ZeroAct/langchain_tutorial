from fastapi import APIRouter
from langchain_core.messages import HumanMessage

from .service import model_service

router = APIRouter(prefix="/model", tags=["model"])


@router.get("/", response_model=list[str])
async def get_model_list():
    return await model_service.get_model_list()


@router.post("/{model}/invoke", response_model=str)
async def invoke(model: str, messages: list[HumanMessage]):
    return await model_service.invoke(model, messages)


@router.post("/{model}/embed", response_model=list[float])
async def embed(model: str, content: str):
    return await model_service.embed(model, content)
