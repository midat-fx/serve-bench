"""Per-request timing records and run-level summaries.

Definitions (same as vLLM's benchmark_serving, so numbers are comparable):
  TTFT  send -> first non-empty text chunk
  TPOT  (e2e - ttft) / (output_tokens - 1), per request
  ITL   gap between consecutive text chunks, pooled over all requests
  E2E   send -> last byte of the stream
"""

from dataclasses import dataclass, field


@dataclass
class Request:
    idx: int
    t_send: float  # seconds since run start
    prompt_tokens: int | None = None  # from server usage, if reported
    cached_tokens: int | None = None  # prompt tokens served from the server's prefix cache
    output_tokens: int = 0
    chunk_times: list[float] = field(default_factory=list)  # since run start
    t_end: float | None = None
    error: str | None = None
    # llama-server's own timings from its final chunk: cache_n, prompt_n, prompt_ms, predicted_ms
    server_timings: dict | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and bool(self.chunk_times)

    @property
    def ttft(self) -> float:
        return self.chunk_times[0] - self.t_send

    @property
    def e2e(self) -> float:
        return self.t_end - self.t_send

    @property
    def tpot(self) -> float | None:
        if self.output_tokens < 2:
            return None
        return (self.e2e - self.ttft) / (self.output_tokens - 1)

    @property
    def itls(self) -> list[float]:
        t = self.chunk_times
        return [b - a for a, b in zip(t, t[1:])]


def percentile(xs: list[float], p: float) -> float | None:
    """Linear interpolation between closest ranks (numpy's default)."""
    if not xs:
        return None
    s = sorted(xs)
    k = (len(s) - 1) * p / 100
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def _dist(xs: list[float]) -> dict:
    return {
        "mean": sum(xs) / len(xs) if xs else None,
        "p50": percentile(xs, 50),
        "p90": percentile(xs, 90),
        "p99": percentile(xs, 99),
    }


def summarize(reqs: list[Request], slo_ttft: float, slo_tpot: float) -> dict:
    ok = [r for r in reqs if r.ok]
    if not ok:
        return {"n": len(reqs), "n_ok": 0, "errors": sorted({r.error for r in reqs if r.error})}
    start = min(r.t_send for r in reqs)
    duration = max(r.t_end for r in ok) - start
    tpots = [r.tpot for r in ok if r.tpot is not None]
    # A request with a single output token has no TPOT; it passes on TTFT alone.
    good = [r for r in ok if r.ttft <= slo_ttft and (r.tpot is None or r.tpot <= slo_tpot)]
    out_tokens = sum(r.output_tokens for r in ok)
    prompt = [r.prompt_tokens for r in ok if r.prompt_tokens is not None]
    cached = [r.cached_tokens for r in ok if r.cached_tokens is not None]
    prompt_ms = [r.server_timings["prompt_ms"] for r in ok
                 if r.server_timings and "prompt_ms" in r.server_timings]
    return {
        "n": len(reqs),
        "n_ok": len(ok),
        "errors": sorted({r.error for r in reqs if r.error}),
        "duration_s": duration,
        "req_per_s": len(ok) / duration,
        "output_tok_per_s": out_tokens / duration,
        "mean_prompt_tokens": sum(prompt) / len(prompt) if prompt else None,
        "mean_cached_tokens": sum(cached) / len(cached) if cached else None,
        "mean_output_tokens": out_tokens / len(ok),
        "mean_server_prompt_ms": sum(prompt_ms) / len(prompt_ms) if prompt_ms else None,
        "ttft_s": _dist([r.ttft for r in ok]),
        "tpot_s": _dist(tpots),
        "itl_s": _dist([x for r in ok for x in r.itls]),
        "e2e_s": _dist([r.e2e for r in ok]),
        "slo": {"ttft_s": slo_ttft, "tpot_s": slo_tpot},
        "goodput_req_per_s": len(good) / duration,
        "slo_attainment": len(good) / len(reqs),
    }
