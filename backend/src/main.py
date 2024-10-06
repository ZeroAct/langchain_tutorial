from chat.router import router as chat_router
from fastapi import FastAPI
from model.router import router as model_router

app = FastAPI()

app.include_router(model_router)
app.include_router(chat_router)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8999)
