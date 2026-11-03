"""Deterministic fake OpenAI /v1/completions streaming server (stdlib only).

Timing model: a request waits for one of `slots` free slots, then sleeps
prefill_s_per_token * prompt_words before the first token and decode_s between
tokens. Tests use it to check that the client measures known latencies.
`server.timings` records what the server itself saw per request (TTFT and token gaps), so tests
can compare the client with the server rather than with nominal sleeps that a slow machine
overshoots.
"""

import argparse
import asyncio
import json
import time


async def serve(host: str, port: int, slots: int, prefill_s_per_token: float, decode_s: float):
    sem = asyncio.Semaphore(slots)
    timings = []

    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        try:
            head = await reader.readuntil(b"\r\n\r\n")
            length = 0
            for line in head.decode().split("\r\n"):
                if line.lower().startswith("content-length:"):
                    length = int(line.split(":")[1])
            req = json.loads(await reader.readexactly(length))
            t0, writes = time.perf_counter(), []
            n_prompt = len(req["prompt"].split())
            n_out = req.get("max_tokens", 16)
            writer.write(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n"
                         b"Connection: close\r\n\r\n")
            await writer.drain()
            async with sem:
                await asyncio.sleep(prefill_s_per_token * n_prompt)
                for i in range(n_out):
                    if i:
                        await asyncio.sleep(decode_s)
                    chunk = {"choices": [{"index": 0, "text": f" t{i}"}]}
                    writer.write(f"data: {json.dumps(chunk)}\n\n".encode())
                    await writer.drain()
                    writes.append(time.perf_counter())
            gaps = [b - a for a, b in zip(writes, writes[1:])]
            timings.append({"ttft_s": writes[0] - t0, "gaps_s": gaps})
            usage = {"prompt_tokens": n_prompt, "completion_tokens": n_out}
            writer.write(f"data: {json.dumps({'choices': [], 'usage': usage})}\n\n".encode())
            writer.write(b"data: [DONE]\n\n")
            await writer.drain()
        except (asyncio.IncompleteReadError, ConnectionError):
            pass
        finally:
            writer.close()

    server = await asyncio.start_server(handle, host, port, backlog=1024)
    server.timings = timings
    return server


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--slots", type=int, default=4)
    ap.add_argument("--prefill-s-per-token", type=float, default=0.0001)
    ap.add_argument("--decode-s", type=float, default=0.01)
    a = ap.parse_args()

    async def go():
        s = await serve("127.0.0.1", a.port, a.slots, a.prefill_s_per_token, a.decode_s)
        async with s:
            await s.serve_forever()

    asyncio.run(go())


if __name__ == "__main__":
    main()
