import uuid

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph
from model.service import model_service
from typing_extensions import TypedDict


class ChatState(TypedDict):
    thread_id: str
    model: str
    system: str
    chat_history: list[BaseMessage]


class ChatService:
    def __init__(self):
        self.states: dict[str, ChatState] = {}

    async def get_model_list(self) -> list[str]:
        return await model_service.get_model_list()

    async def get(self, thread_id: str):
        return self.states.get(thread_id, None)

    async def get_thread_list(self) -> list[str]:
        return list(self.states.keys())

    async def create(
        self,
        model: str = None,
        system: str = "assistant",
        chat_history: list[BaseMessage] = [],
        thread_id: str = None,
    ) -> ChatState:
        thread_id = thread_id or str(uuid.uuid4())
        if thread_id not in self.states:
            model_list = await self.get_model_list()
            model = model or model_list[0]
            if model not in model_list:
                raise ValueError(f"Model {model} not found. Available models: {model_list}")
            self.states[thread_id] = {
                "thread_id": thread_id,
                "model": model,
                "system": system,
                "chat_history": chat_history,
            }
        return self.states[thread_id]

    async def update(
        self,
        thread_id: str,
        model: str = None,
        system: str = None,
        chat_history: list[BaseMessage] = None,
    ) -> ChatState:
        if thread_id not in self.states:
            raise ValueError(f"Thread {thread_id} not found")

        state = self.states[thread_id]
        if model:
            state["model"] = model
        if system:
            state["system"] = system
        if chat_history:
            state["chat_history"] = chat_history
        return state

    async def chat(
        self,
        thread_id: str,
        input: str,
    ) -> str:
        if thread_id not in self.states:
            state: ChatState = self.create(model=None, thread_id=thread_id)
        else:
            state: ChatState = self.states[thread_id]

        try:
            messages = [
                SystemMessage(content=state["system"]),
                *state["chat_history"],
                HumanMessage(content=input),
            ]
            result = await model_service.invoke(model=state["model"], messages=messages)
            state["chat_history"].append(HumanMessage(content=input))
            state["chat_history"].append(AIMessage(content=result))
            return result
        except Exception as e:
            raise ValueError("Failed to invoke model") from e

    async def delete(self, thread_id: str = None):
        if thread_id in self.states:
            del self.states[thread_id]
        return True


chat_service = ChatService()


if __name__ == "__main__":
    import asyncio

    async def main():
        print(await chat_service.get_model_list())
        print(await chat_service.get_thread_list())
        print(await chat_service.create(thread_id="test"))
        print(await chat_service.get("test"))
        print(await chat_service.update("test", system="chatbot"))
        print("Me: Hello")
        print(await chat_service.chat("test", "Hello"))
        print("Me: I'm huijae")
        print(await chat_service.chat("test", "I'm huijae"))
        print("Me: Who am I?")
        print(await chat_service.chat("test", "Who am I?"))
        print("Me: I want you to help me code in python. I need calculator")
        print(
            await chat_service.chat(
                "test", "I want you to help me code in python. I need calculator"
            )
        )
        print("Me: Can you say my name?")
        print(await chat_service.chat("test", "Can you say my name?"))

    asyncio.run(main())
