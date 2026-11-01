import argparse
import asyncio
import json
import pathlib

from .bench import run
from .report import report


def cmd_run(a):
    cfg = {k: getattr(a, k) for k in (
        "url", "model", "label", "pattern", "n", "concurrency", "rate", "burst", "prompt_tokens",
        "shared_prefix", "max_tokens", "ignore_eos", "seed", "warmup", "slo_ttft", "slo_tpot")}
    cfg["extra_body"] = json.loads(a.extra_body)
    cfg["tags"] = dict(t.split("=", 1) for t in a.tag)
    res = asyncio.run(run(cfg))
    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    s = res["summary"]
    if not s["n_ok"]:
        raise SystemExit(f"all {s['n']} requests failed: {s['errors']}")
    print(f"{a.label}: ok {s['n_ok']}/{s['n']}  ttft p50 {s['ttft_s']['p50'] * 1e3:.0f}ms "
          f"p99 {s['ttft_s']['p99'] * 1e3:.0f}ms  tpot p50 {s['tpot_s']['p50'] * 1e3:.1f}ms  "
          f"{s['output_tok_per_s']:.0f} tok/s  goodput {s['goodput_req_per_s']:.2f} req/s  "
          f"load {res['state_before']['loadavg'][0]}")


def main():
    ap = argparse.ArgumentParser(prog="servebench")
    sub = ap.add_subparsers(required=True)

    r = sub.add_parser("run", help="one benchmark run -> one results JSON")
    r.add_argument("--url", required=True, help="server base URL, e.g. http://127.0.0.1:8081")
    r.add_argument("--model", default="default")
    r.add_argument("--label", required=True, help="server/engine name used in plots")
    r.add_argument("--out", required=True)
    r.add_argument("--pattern", choices=["closed", "constant", "poisson", "burst"], default="closed")
    r.add_argument("--n", type=int, default=32, help="measured requests")
    r.add_argument("--concurrency", type=int, default=1, help="closed loop: users in flight")
    r.add_argument("--rate", type=float, default=1.0, help="open loop: mean requests/s")
    r.add_argument("--burst", type=int, default=8, help="burst pattern: requests per burst")
    r.add_argument("--prompt-tokens", type=int, default=256)
    r.add_argument("--shared-prefix", type=int, default=0, help="words shared by all prompts")
    r.add_argument("--max-tokens", type=int, default=128)
    r.add_argument("--ignore-eos", action="store_true")
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--warmup", type=int, default=2)
    r.add_argument("--slo-ttft", type=float, default=1.0, help="seconds")
    r.add_argument("--slo-tpot", type=float, default=0.05, help="seconds")
    r.add_argument("--extra-body", default="{}", help="JSON merged into each request body")
    r.add_argument("--tag", action="append", default=[], help="k=v stored with the run")
    r.set_defaults(fn=cmd_run)

    p = sub.add_parser("report", help="results/SUMMARY.md and plots from all result files")
    p.add_argument("--results", default="results")
    p.add_argument("--img", default="docs/img")
    p.set_defaults(fn=lambda a: print(*report(pathlib.Path(a.results), pathlib.Path(a.img))))

    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
