"""
VitalLens FastAPI backend.

Start with:
  uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000

WebSocket:  ws://localhost:8000/ws/vitals?camera=1
REST:       POST http://localhost:8000/session/end
Health:     GET  http://localhost:8000/health
"""
from __future__ import annotations

import asyncio
import json
import time
import threading
from typing import Optional

import os
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from backend.inference import VitalsEngine, DEFAULT_CAMERA_INDEX, get_rppg_config, update_rppg_config

_cors_origins = [o.strip() for o in os.environ.get("CORS_ORIGINS", "*").split(",")]

app = FastAPI(title="VitalLens API", version="1.0.0")

# Most-recent active engine — used by /video_feed
_engine_lock   = threading.Lock()
_active_engine: VitalsEngine | None = None

def _set_active_engine(eng: VitalsEngine | None) -> None:
    global _active_engine
    with _engine_lock:
        _active_engine = eng

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok", "timestamp": time.time()}


@app.get("/cameras")
async def list_cameras():
    """Returns available camera indices. Frontend uses this to populate the camera picker."""
    loop    = asyncio.get_event_loop()
    cameras = await loop.run_in_executor(None, VitalsEngine.enumerate_cameras)
    return cameras


@app.get("/video_feed")
async def video_feed():
    """MJPEG stream of the active camera — lets the frontend display video
    without opening the camera a second time via getUserMedia."""
    boundary = b"--frame"

    async def generate():
        while True:
            with _engine_lock:
                eng = _active_engine
            if eng is None:
                await asyncio.sleep(0.1)
                continue
            jpeg = eng.get_jpeg()
            if jpeg:
                yield (
                    boundary + b"\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" +
                    jpeg + b"\r\n"
                )
            await asyncio.sleep(1 / 30)  # ~30 fps

    return StreamingResponse(
        generate(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


# ---------------------------------------------------------------------------
# WebSocket — real-time vitals stream
# ---------------------------------------------------------------------------

@app.websocket("/ws/vitals")
async def ws_vitals(
    websocket:     WebSocket,
    camera:        int   = Query(default=DEFAULT_CAMERA_INDEX, description="Camera device index"),
    camera_url:    str   = Query(default=None,  description="MJPEG stream URL; overrides camera index"),
    source:        str   = Query(default="camera", description="'camera' = local/MJPEG capture; 'browser' = client streams frames"),
    lock_exposure: float = Query(default=None,  description="Fixed exposure value; local cameras only"),
    model:         str   = Query(default=None,  description="'factorizephys' | 'efficientphys' — overrides server default"),
):
    """
    Vitals WebSocket.  Two modes:

    camera mode  (source=camera, default):
        Backend opens the camera (local device or MJPEG URL) and streams
        vitals JSON to the client every ~1 s.

    browser mode (source=browser):
        Client sends raw JPEG frames as binary WebSocket messages.
        Backend processes them and streams vitals JSON back every ~1 s.
        Used for cloud deployments where the backend has no physical camera.
    """
    await websocket.accept()

    engine = VitalsEngine(
        camera_index=camera, camera_url=camera_url,
        brightness_norm=True, lock_exposure=lock_exposure,
        model=model,
    )
    loop = asyncio.get_event_loop()

    try:
        if source == "browser":
            await loop.run_in_executor(None, engine.start_browser_mode)
            _set_active_engine(engine)

            async def _send_vitals():
                try:
                    while engine.is_running:
                        await websocket.send_text(json.dumps(engine.get_message()))
                        await asyncio.sleep(1.0)
                except Exception:
                    pass

            send_task    = asyncio.create_task(_send_vitals())
            process_task = None
            try:
                while True:
                    data = await websocket.receive_bytes()
                    if process_task is None or process_task.done():
                        process_task = asyncio.ensure_future(
                            loop.run_in_executor(None, engine.push_frame, data)
                        )
            except WebSocketDisconnect:
                pass
            except Exception as e:
                print(f"Browser WS error: {e}")
            finally:
                send_task.cancel()

        else:
            await loop.run_in_executor(None, engine.start)
            _set_active_engine(engine)

            while engine.is_running:
                await websocket.send_text(json.dumps(engine.get_message()))
                await asyncio.sleep(1.0)

    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"WebSocket error: {e}")
        try:
            await websocket.send_text(json.dumps({"error": str(e)}))
        except Exception:
            pass
    finally:
        _set_active_engine(None)
        engine.stop()


# ---------------------------------------------------------------------------
# Runtime config — tune signal-processing thresholds without restarting
# ---------------------------------------------------------------------------

@app.get("/session/config")
def get_session_config():
    return get_rppg_config()

@app.patch("/session/config")
async def patch_session_config(updates: Optional[dict] = Body(default=None)):
    if not updates:
        return get_rppg_config()
    return update_rppg_config(updates)


# ---------------------------------------------------------------------------
# Age estimation + HR zones
# ---------------------------------------------------------------------------

@app.post("/session/estimate-age")
async def estimate_age(payload: Optional[dict] = Body(default=None)):
    """
    Payload: { "image": "<base64-encoded JPEG of face crop>" }
    Returns: { "age": int, "hr_zones": { "1": [lo, hi], ... } }
    One call per session — used to personalise HR zone display.
    """
    if not payload or not payload.get("image"):
        return {"age": None, "hr_zones": None}
    try:
        from openai import OpenAI
        client   = OpenAI()   # reads OPENAI_API_KEY; raises if missing
        img_data = payload["image"]
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=10,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{img_data}",
                            "detail": "low",
                        },
                    },
                    {
                        "type": "text",
                        "text": "Estimate the age of the person in this photo. Reply with a single integer only, nothing else.",
                    },
                ],
            }],
        )
        age    = int(response.choices[0].message.content.strip())
        max_hr = 220 - age
        hr_zones = {
            "1": [int(max_hr * 0.50), int(max_hr * 0.60)],
            "2": [int(max_hr * 0.60), int(max_hr * 0.70)],
            "3": [int(max_hr * 0.70), int(max_hr * 0.80)],
            "4": [int(max_hr * 0.80), int(max_hr * 0.90)],
            "5": [int(max_hr * 0.90), max_hr],
        }
        print(f"Age estimate: {age} yrs → max HR {max_hr} BPM")
        return {"age": age, "hr_zones": hr_zones}
    except Exception as e:
        print(f"Age estimation failed: {e}")
        return {"age": None, "hr_zones": None}


# ---------------------------------------------------------------------------
# Session end — returns computed summary from frontend readings
# ---------------------------------------------------------------------------

@app.post("/session/end")
async def session_end(payload: Optional[dict] = Body(default=None)):
    """
    Accepts session readings from the frontend and returns a full summary.
    Response matches the contract in docs/05_backend_contract_and_reference.md.
    If no readings, returns an empty response — frontend falls back to client-side computation.
    """
    if not payload or not payload.get("readings"):
        return {}

    readings = payload["readings"]

    hrs      = [r["hr"]     for r in readings if r.get("hr",     0) > 0]
    brs      = [r["br"]     for r in readings if r.get("br",     0) > 0]
    hrvs     = [r["hrv"]    for r in readings if r.get("hrv",    0) > 0]
    stresses = [r["stress"] for r in readings if r.get("stress") is not None]

    def safe_avg(lst): return round(sum(lst) / len(lst), 1) if lst else 0.0
    def safe_min(lst): return round(min(lst), 1) if lst else 0.0
    def safe_max(lst): return round(max(lst), 1) if lst else 0.0

    lightings = [r.get("lighting", "Good") for r in readings]
    total     = len(lightings)
    breakdown = {
        label: round(lightings.count(label) / total, 3) if total else 0.0
        for label in ("Good", "Mixed", "Poor")
    }

    bvp_series = []
    for r in readings:
        bvp = r.get("bvp")
        if bvp:
            bvp_series.extend(bvp)

    hr_series  = [
        {"hr": r["hr"], "timestamp": r["timestamp"]}
        for r in readings if r.get("hr") and r.get("timestamp")
    ]
    timestamps = [r["timestamp"] for r in readings if r.get("timestamp")]

    ts_list = [r["timestamp"] for r in readings if r.get("timestamp")]
    actual_duration = round(ts_list[-1] - ts_list[0]) if len(ts_list) >= 2 else len(readings)

    return {
        "duration_seconds":   actual_duration,
        "avg_hr":             safe_avg(hrs),
        "avg_br":             safe_avg(brs),
        "avg_hrv":            safe_avg(hrvs),
        "avg_stress":         safe_avg(stresses),
        "min_hr":             safe_min(hrs),
        "max_hr":             safe_max(hrs),
        "lighting_breakdown": breakdown,
        "bvp_series":         bvp_series,
        "hr_series":          hr_series,
        "timestamps":         timestamps,
    }
