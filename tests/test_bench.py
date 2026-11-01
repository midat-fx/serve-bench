import asyncio
import statistics

import pytest

import servebench.bench
from servebench.bench import arrivals, make_prompts, run
from servebench.fakeserver import serve
from servebench.metrics import Request, percentile, summarize

servebench.bench.GPU_SAMPLES = 1  # each sample costs 0.2 s


def test_percentile_matches_numpy_linear():
    xs = [1, 2, 3, 4, 10]
    assert percentile(xs, 0) == 1
    assert percentile(xs, 50) == 3
    assert percentile(xs, 90) == pytest.approx(7.6)  # numpy.percentile(xs, 90)
    assert percentile(xs, 100) == 10
    assert percentile([], 50) is None


def test_request_metrics_and_goodput():
    fast = Request(0, t_send=0.0, output_tokens=5, chunk_times=[0.1, 0.2, 0.3, 0.4, 0.5], t_end=0.5)
    slow = Request(1, t_send=1.0, output_tokens=3, chunk_times=[3.0, 3.1, 3.2], t_end=3.2)
    failed = Request(2, t_send=1.5, t_end=1.6, error="http 500")
    assert fast.ttft == pytest.approx(0.1)
    assert fast.tpot == pytest.approx(0.1)  # 0.4s over 4 gaps
    assert slow.ttft == pytest.approx(2.0)
    s = summarize([fast, slow, failed], slo_ttft=1.0, slo_tpot=0.2)
    assert s["n_ok"] == 2 and s["errors"] == ["http 500"]
    assert s["duration_s"] == pytest.approx(3.2)
    assert s["output_tok_per_s"] == pytest.approx(8 / 3.2)
    assert s["goodput_req_per_s"] == pytest.approx(1 / 3.2)  # only `fast` meets the TTFT SLO
    assert s["slo_attainment"] == pytest.approx(1 / 3)  # failed requests count against the SLO
    assert s["itl_s"]["p50"] == pytest.approx(0.1)


def test_tpot_uses_reported_tokens_when_chunks_pack_several():
    r = Request(0, t_send=0.0, output_tokens=9, chunk_times=[0.1, 0.5, 0.9], t_end=0.9)
    assert r.tpot == pytest.approx(0.1)  # 0.8s over 8 token gaps, not 2 chunk gaps


def test_arrival_patterns_share_mean_rate():
    n, rate = 4000, 5.0
    const = arrivals("constant", n, rate)
    assert const[:3] == [0.0, 0.2, 0.4]
    pois = arrivals("poisson", n, rate, seed=1)
    gaps = [b - a for a, b in zip(pois, pois[1:])]
    assert statistics.mean(gaps) == pytest.approx(1 / rate, rel=0.05)
    assert statistics.stdev(gaps) == pytest.approx(1 / rate, rel=0.1)  # exponential: sd == mean
    assert pois == arrivals("poisson", n, rate, seed=1)
    burst = arrivals("burst", 8, 2.0, burst=4)
    assert burst == [0, 0, 0, 0, 2, 2, 2, 2]


def test_prompts_share_exactly_the_prefix():
    ps = make_prompts(3, tokens=50, shared_prefix=30, seed=0, nonce="abc")
    head = ps[0].split("\n")[0]
    assert len(head.split()) == 31  # nonce + 30 words
    assert all(p.startswith(head + "\n") for p in ps)
    assert len({p.split("\n")[1] for p in ps}) == 3
    unique = make_prompts(3, tokens=50, shared_prefix=0, seed=0, nonce="abc")
    assert len({p.split()[0] for p in unique}) == 3  # no common first token
    assert all(len(p.split()) == 51 for p in unique)


def _cfg(port, **kw):
    cfg = dict(url=f"http://127.0.0.1:{port}", model="fake", label="fake", pattern="closed", n=8,
               concurrency=1, rate=1.0, burst=1, prompt_tokens=100, shared_prefix=0, max_tokens=21,
               ignore_eos=False, seed=0, warmup=1, slo_ttft=1.0, slo_tpot=0.05, extra_body={})
    return cfg | kw


async def _with_fake(cfg_kw, slots=1, prefill=0.0005, decode=0.01):
    server = await serve("127.0.0.1", 0, slots, prefill, decode)
    port = server.sockets[0].getsockname()[1]
    async with server:
        return await run(_cfg(port, **cfg_kw))


def test_end_to_end_measures_known_latencies():
    res = asyncio.run(_with_fake({}))
    s = res["summary"]
    assert s["n_ok"] == 8
    assert s["mean_prompt_tokens"] == 101  # nonce word + 100
    assert s["mean_output_tokens"] == 21
    # fake prefill = 101 * 0.5ms ~= 50ms, decode 10ms/token; allow scheduler jitter
    assert s["ttft_s"]["p50"] == pytest.approx(0.0505, abs=0.015)
    assert s["tpot_s"]["p50"] == pytest.approx(0.010, abs=0.003)
    assert res["state_before"]["loadavg"] and res["machine"]["cpu"]


def test_queueing_shows_up_in_ttft_not_tpot():
    # 4 users, 1 server slot: each request waits for ~3 others (~0.25s each) before prefill.
    res = asyncio.run(_with_fake({"concurrency": 4, "n": 12, "slo_ttft": 0.5}))
    s = res["summary"]
    assert s["ttft_s"]["p50"] == pytest.approx(0.8, abs=0.1)
    assert s["tpot_s"]["p50"] == pytest.approx(0.010, abs=0.003)
    # Only the first two requests start before the queue fills (TTFT ~0.05s, ~0.3s).
    assert s["slo_attainment"] == pytest.approx(2 / 12)


def test_report_builds_table_and_plots(tmp_path):
    import json

    from servebench.report import report

    for exp, xs in {"concurrency": [1, 2], "rate": [1, 2], "arrivals": ["constant", "burst"],
                    "prompt": [50, 100], "prefix": [0, 50]}.items():
        for server in ("llamacpp", "mlx"):
            for x in xs:
                kw = {"tags": {"experiment": exp, "x": str(x)}, "label": server, "n": 3}
                kw |= {"concurrency": x} if exp == "concurrency" else {}
                res = asyncio.run(_with_fake(kw, slots=2, decode=0.001))
                d = tmp_path / "results" / exp
                d.mkdir(parents=True, exist_ok=True)
                (d / f"{server}_{x}.json").write_text(json.dumps(res))
    pngs = report(tmp_path / "results", tmp_path / "img")
    assert sorted(pngs) == ["arrivals.png", "concurrency.png", "prefix.png", "prompt.png", "rate.png"]
    summary = (tmp_path / "results" / "SUMMARY.md").read_text()
    assert summary.count("| llamacpp |") == 10 and "`arrivals/mlx_burst`" in summary
