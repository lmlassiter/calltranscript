# Real-Time Call Transcription Demo (Whisper + Twilio)

This is a demo project I made a few months ago to showcase the performance of locally run whisper to transcribe live phone conversations on twilio.

The transcriptions are not perfect and adjustments are needed - chunk sizing and segment merging would benefit from further tuning, especially for overlapping or partial speech.


I have included the py file doing all the heavy lifting. It's a WebSocket server (FastAPI) handling audio decoding, resampling, and transcription.

The frontend is just a functional mock interface designed to demonstrate the backends' real-time AI pipeline, not a fully built out front end. The demo video is in the demo folder

If you’d like to see the frontend work, I’m happy to clean and share those files — or walk through a separate project with a fully functional UI.


Frontend: Next.js 15 / TypeScript / Tailwind
Backend: FastAPI / SSE / Whisper / Librosa / Torch / Twilio Voice API