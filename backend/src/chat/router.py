from fastapi import APIRouter
from langchain_core.messages import HumanMessage

from .service import chat_service

router = APIRouter(prefix="/chat", tags=["chat"])


@router.get("/model", response_model=list[str])
async def get_model_list():
    return await chat_service.get_model_list()


@router.get("/", response_model=list[str])
async def get_thread_list():
    return await chat_service.get_thread_list()


@router.post("/", response_model=dict)
async def create(
    model: str = None,
    system: str = "assistant",
    chat_history: list[HumanMessage] = [],
    thread_id: str = None,
):
    return await chat_service.create(
        model=model, system=system, chat_history=chat_history, thread_id=thread_id
    )


@router.get("/{thread_id}", response_model=dict)
async def get(thread_id: str):
    return await chat_service.get(thread_id)


@router.post("/{thread_id}", response_model=str)
async def chat(thread_id: str, input: str):
    return await chat_service.chat(thread_id=thread_id, input=input)


@router.put("/{thread_id}", response_model=dict)
async def update(
    thread_id: str, model: str = None, system: str = None, chat_history: list[HumanMessage] = None
):
    return await chat_service.update(
        thread_id, model=model, system=system, chat_history=chat_history
    )


@router.delete("/{thread_id}", response_model=bool)
async def delete(thread_id: str):
    return await chat_service.delete(thread_id)
