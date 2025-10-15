import asyncio
from typing import Set
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import os

PAGES_ORIGIN = os.environ.get("PAGES_ORIGIN", "*")  # 允许的前端来源，先 * ，上线可填你的 pages 域名

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=[PAGES_ORIGIN] if PAGES_ORIGIN != "*" else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Frame(BaseModel):
    t: int
    alpha: float
    driverA: str
    driverB: str
    features: dict   # {"d_head":float,"d_brake":float,"d_thr":float}
    sim: dict        # {"NOR":..,"RUS":..,"VER":..,"HYB":..}

clients: Set[WebSocket] = set()
queue: "asyncio.Queue[dict]" = asyncio.Queue()

@app.websocket("/ws")
async def ws(ws: WebSocket):
    await ws.accept()
    clients.add(ws)
    try:
        while True:
            msg = await queue.get()
            await ws.send_json(msg)
    except WebSocketDisconnect:
        pass
    finally:
        clients.discard(ws)

@app.post("/push")
async def push(frame: Frame):
    await queue.put(frame.dict())
    return {"ok": True}

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), reload=False)
