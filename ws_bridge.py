# ws_bridge.py
import json
from typing import Dict, Set

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

app = FastAPI()
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

clients: Set[WebSocket] = set()

@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    clients.add(ws)
    try:
        while True:
            # 可选：接收前端消息，这里忽略
            await ws.receive_text()
    except Exception:
        pass
    finally:
        clients.discard(ws)

@app.post("/broadcast")
async def broadcast(payload: Dict):
    dead = []
    text = json.dumps(payload)
    for ws in list(clients):
        try:
            await ws.send_text(text)
        except Exception:
            dead.append(ws)
    for ws in dead:
        clients.discard(ws)
    return {"ok": True, "sent": len(clients)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8765)
