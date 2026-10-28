"""Progress streams live; only verified answer tokens are released."""

import json
from fastapi.responses import StreamingResponse


def sse_stream(session, question):
    async def events():
        try:
            async for event in session.astream(question):
                yield f"data: {json.dumps(event, default=str)}\n\n".encode()
        except Exception:
            yield b'data: {"event":"error","message":"Stream unavailable"}\n\n'
        yield b'data: {"event":"done"}\n\n'

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
    )
