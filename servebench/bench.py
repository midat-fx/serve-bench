"""Load generator: prompt construction, arrival schedules, streaming client, run loop."""

import asyncio
import json
import math
import os
import platform
import random
import secrets
import subprocess
import time
from dataclasses import asdict

import httpx

from .metrics import Request, summarize

# Common single-token English words, so prompt length in tokens ~= word count.
# The real count is taken from the server's usage field and reported, never assumed.
WORDS = (
    "the of and to in is was for on that with as by at from his her it be this are an or had not "
    "but have one were all which their can there more when has been who out so will no if about "
    "into time up other than only some could them two first then may these its would over new "
    "after like most made such also where many before through back years must much how even down "
    "well way because each work life long day great own old world see between still last high "
    "might part under same few city water place small house large found never hand next "
    "light point food night state line home river group green road open early hill north south "
    "west east plant bird stone glass paper music table window market garden summer winter"
).split()


def lengths(n: int, mean: int, sigma: float, seed: str) -> list[int]:
    """Per-request lengths: all `mean` when sigma is 0, else lognormal with that mean
    (mu = ln(mean) - sigma^2 / 2), rounded and at least 1."""
    if not sigma:
        return [mean] * n
    rng = random.Random(seed)
    mu = math.log(mean) - sigma**2 / 2
    return [max(1, round(rng.lognormvariate(mu, sigma))) for _ in range(n)]


def make_prompts(n: int, tokens: int | list[int], shared_prefix: int, seed: int,
                 nonce: str) -> list[str]:
    """n prompts of ~`tokens` words (one length, or one per prompt). The first `shared_prefix`
    words are identical across prompts; the rest are unique per request. `nonce` keeps runs
    from hitting a server's prompt cache left over from an earlier run."""
    if isinstance(tokens, int):
        tokens = [tokens] * n
    rng = random.Random(f"{seed}-prefix")
    prefix = f"{nonce} " + " ".join(rng.choice(WORDS) for _ in range(shared_prefix))
    out = []
    for i in range(n):
        rng = random.Random(f"{seed}-{i}")
        body = " ".join(rng.choice(WORDS) for _ in range(max(tokens[i] - shared_prefix, 1)))
        out.append(f"{prefix}\n{i}: {body}" if shared_prefix else f"{nonce}-{i}: {body}")
    return out


def arrivals(pattern: str, n: int, rate: float, burst: int = 1, seed: int = 0) -> list[float]:
    """Send offsets in seconds. All open-loop patterns have the same mean rate, so they
    differ only in how bunched the arrivals are."""
    if pattern == "constant":
        return [i / rate for i in range(n)]
    if pattern == "poisson":
        rng = random.Random(seed)
        t, out = 0.0, []
        for _ in range(n):
            out.append(t)
            t += rng.expovariate(rate)
        return out
    if pattern == "burst":
        return [(i // burst) * burst / rate for i in range(n)]
    raise ValueError(f"unknown pattern {pattern!r}")


TIMING_KEYS = ("cache_n", "prompt_n", "prompt_ms", "predicted_n", "predicted_ms")


async def send(client: httpx.AsyncClient, url: str, body: dict, req: Request, t0: float):
    clock = time.perf_counter
    req.t_send = clock() - t0
    chunks, usage_out = 0, None
    path = "/v1/chat/completions" if "messages" in body else "/v1/completions"
    try:
        async with client.stream("POST", f"{url}{path}", json=body) as r:
            if r.status_code != 200:
                await r.aread()
                req.error = f"http {r.status_code}"
                return
            async for line in r.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                obj = json.loads(data)
                if usage := obj.get("usage"):
                    req.prompt_tokens = usage.get("prompt_tokens")
                    req.cached_tokens = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
                    usage_out = usage.get("completion_tokens")
                if t := obj.get("timings"):  # llama-server only
                    req.server_timings = {k: t[k] for k in TIMING_KEYS if k in t}
                for ch in obj.get("choices") or []:
                    delta = ch.get("delta") or {}
                    if (delta.get("reasoning_content") or "").strip():
                        # Thinking left on: TTFT would time the reasoning, not the answer. LM Studio
                        # MLX sends an empty "\n\n" think block even with /no_think.
                        req.error = "reasoning output"
                        return
                    if ch.get("text") or delta.get("content"):
                        req.chunk_times.append(clock() - t0)
                        chunks += 1
    except (httpx.HTTPError, json.JSONDecodeError) as e:
        req.error = type(e).__name__
    finally:
        req.t_end = clock() - t0
        # Servers may pack several tokens into one chunk; usage is the source of truth.
        req.output_tokens = usage_out or chunks
        if not req.chunk_times and not req.error:
            req.error = "empty stream"


def _sh(cmd: str) -> str:
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=10).stdout.strip()
    except subprocess.SubprocessError:
        return ""


GPU_SAMPLES = 5


def gpu_util() -> float | None:
    """Mean GPU 'Device Utilization %' from IOAccelerator over ~1 s (no sudo needed).
    Load average misses GPU contention, e.g. another app rendering on the GPU."""
    vals = []
    for _ in range(GPU_SAMPLES):
        out = _sh("ioreg -r -d 1 -c IOAccelerator | grep -o '\"Device Utilization %\"=[0-9]*'")
        vals += [int(v.split("=")[1]) for v in out.splitlines()]
        time.sleep(0.2)
    return sum(vals) / len(vals) if vals else None


def machine_state() -> dict:
    """What else the machine was doing: load, GPU use, free memory, power, top CPU users."""
    free = _sh("memory_pressure -Q | awk -F': ' '/percentage/{print $2}'")
    return {
        "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "loadavg": [round(x, 2) for x in os.getloadavg()],
        "gpu_util_pct": gpu_util(),
        "mem_free_pct": free,
        "power": _sh("pmset -g batt | head -1 | sed \"s/Now drawing from //;s/'//g\""),
        # Command basenames only (-c): no paths or arguments leak into results.
        "top_cpu": _sh("ps -Aceo pcpu,comm -r | sed -n 2,6p").splitlines(),
    }


def machine_info() -> dict:
    return {
        "cpu": _sh("sysctl -n machdep.cpu.brand_string"),
        "mem_gb": round(int(_sh("sysctl -n hw.memsize") or 0) / 2**30),
        "os": f"macOS {platform.mac_ver()[0]}",
        "python": platform.python_version(),
        "httpx": httpx.__version__,
        "servebench_commit": _sh("git rev-parse --short HEAD 2>/dev/null"),
    }


async def run(cfg: dict) -> dict:
    """cfg keys: url model pattern n concurrency rate burst prompt_tokens shared_prefix
    max_tokens ignore_eos seed warmup slo_ttft slo_tpot extra_body, and optionally
    endpoint (completions|chat), prompt_sigma, output_sigma (lognormal lengths)"""
    nonce = secrets.token_hex(4)
    total = cfg["n"] + cfg["warmup"]
    plen = lengths(total, cfg["prompt_tokens"], cfg.get("prompt_sigma", 0), f"{cfg['seed']}-plen")
    outs = lengths(total, cfg["max_tokens"], cfg.get("output_sigma", 0), f"{cfg['seed']}-olen")
    prompts = list(zip(make_prompts(total, plen, cfg["shared_prefix"], cfg["seed"], nonce), outs))
    warm, prompts = prompts[: cfg["warmup"]], prompts[cfg["warmup"]:]

    def body(po):
        p, max_tokens = po
        text = {"messages": [{"role": "user", "content": p + cfg.get("user_suffix", "")}]} if cfg.get("endpoint") == "chat" \
            else {"prompt": p}
        b = {"model": cfg["model"], **text, "max_tokens": max_tokens,
             "temperature": 0, "stream": True, "stream_options": {"include_usage": True}}
        if cfg["ignore_eos"]:
            b["ignore_eos"] = True
        return b | cfg.get("extra_body", {})

    limits = httpx.Limits(max_connections=1000, max_keepalive_connections=1000)
    async with httpx.AsyncClient(timeout=httpx.Timeout(600.0), limits=limits) as client:
        before = machine_state()  # before warmup, so our own GPU work is not in it
        for p in warm:  # model load, Metal shader compile, connection setup
            await send(client, cfg["url"], body(p), Request(-1, 0.0), time.perf_counter())
        reqs = [Request(i, 0.0) for i in range(cfg["n"])]
        t0 = time.perf_counter()
        if cfg["pattern"] == "closed":
            queue = iter(range(cfg["n"]))

            async def worker():
                for i in queue:
                    await send(client, cfg["url"], body(prompts[i]), reqs[i], t0)

            await asyncio.gather(*(worker() for _ in range(cfg["concurrency"])))
        else:
            offsets = arrivals(cfg["pattern"], cfg["n"], cfg["rate"], cfg["burst"], cfg["seed"])

            async def fire(i):
                await asyncio.sleep(max(0.0, t0 + offsets[i] - time.perf_counter()))
                await send(client, cfg["url"], body(prompts[i]), reqs[i], t0)

            await asyncio.gather(*(fire(i) for i in range(cfg["n"])))
        after = machine_state()

    return {
        "config": cfg | {"nonce": nonce},
        "machine": machine_info(),
        "state_before": before,
        "state_after": after,
        "summary": summarize(reqs, cfg["slo_ttft"], cfg["slo_tpot"]),
        "requests": [asdict(r) for r in reqs],
    }
