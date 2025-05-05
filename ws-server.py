import asyncio
import os
import base64
import json
import logging
import numpy as np
import librosa
import whisper
import torch
from fastapi import FastAPI, WebSocket, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from twilio.twiml.voice_response import VoiceResponse
from sse_starlette.sse import EventSourceResponse
from queue import Queue
import audioop  # for µ-law decoding

logging.basicConfig(level=logging.INFO)
app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MOBILE_NUMBER = os.getenv("MOBILE_PHONE_NUMBER")
TWILIO_NUMBER = os.getenv("TWILIO_PHONE_NUMBER")
STREAM_URL = os.getenv("TWILIO_STREAM_URL")

# Load the Whisper model onto the GPU
model = whisper.load_model("large-v3-turbo", device="cuda")
logging.info(f"CUDA Available: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0)}")

transcript_queue = Queue()

# Twilio sends 8000Hz µ-law audio (1 byte per sample).
TWILIO_SAMPLE_RATE = 8000
WHISPER_SAMPLE_RATE = 16000
CHUNK_DURATION_SEC = 3  # Adjust for context; longer chunks capture more speech.
CHUNK_SIZE = TWILIO_SAMPLE_RATE * CHUNK_DURATION_SEC  # in bytes (e.g. 8000 * 3 = 24000 bytes)

# Set an energy threshold: if average absolute amplitude is below this, skip transcription.
ENERGY_THRESHOLD = 0.008

@app.post("/voice")
async def voice(request: Request):
    response = VoiceResponse()
    # Instruct Twilio to stream both inbound and outbound audio
    response.start().stream(
        url=STREAM_URL,
        track="both_tracks",
    )
    dial = response.dial(callerId=TWILIO_NUMBER)
    dial.number(MOBILE_NUMBER)
    logging.info("✅ Twilio Voice endpoint hit, returning TwiML.")
    return Response(content=str(response), media_type="application/xml")

@app.websocket("/audio")
async def audio(ws: WebSocket):
    await ws.accept()
    logging.info("✅ Twilio websocket connected.")
    
    buffers = {'inbound': b'', 'outbound': b''}
    
    while True:
        try:
            msg_json = await ws.receive_text()
            msg = json.loads(msg_json)
            if msg.get("event") == "media":
                track = msg["media"]["track"]
                chunk = base64.b64decode(msg["media"]["payload"])
                buffers[track] += chunk

                while len(buffers[track]) >= CHUNK_SIZE:
                    ulaw_bytes = buffers[track][:CHUNK_SIZE]
                    buffers[track] = buffers[track][CHUNK_SIZE:]

                    # Decode µ-law to linear PCM16 (2 bytes per sample)
                    pcm_bytes = audioop.ulaw2lin(ulaw_bytes, 2)
                    audio_np = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                    logging.info(f"Audio stats {track}: len={len(audio_np)}, min={audio_np.min():.3f}, max={audio_np.max():.3f}, mean={audio_np.mean():.3f}")

                    # Skip this chunk if energy is very low (likely muted)
                    if np.abs(audio_np).mean() < ENERGY_THRESHOLD:
                        logging.info(f"Skipping {track} chunk due to low energy.")
                        continue

                    # Resample from 8000Hz to 16000Hz
                    audio_resampled = librosa.resample(audio_np, orig_sr=TWILIO_SAMPLE_RATE, target_sr=WHISPER_SAMPLE_RATE)
                    audio_segment = whisper.pad_or_trim(audio_resampled)

                    result = model.transcribe(
                        audio_segment,
                        fp16=False,
                        language="en",
                        condition_on_previous_text=False
                    )
                    logging.info(f"✅ {track} transcript: {result['text']}")
                    transcript_queue.put({"direction": track, "text": result["text"]})
        except Exception as e:
            logging.error(f"Websocket error: {e}")
            break

@app.get("/stream")
async def stream_transcript():
    async def event_gen():
        while True:
            if transcript_queue.empty():
                await asyncio.sleep(0.1)
                continue
            yield {"data": json.dumps(transcript_queue.get())}
    # Serve locally (ngrok is only used for /voice and /audio)
    return EventSourceResponse(event_gen())
