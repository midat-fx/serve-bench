#!/usr/bin/env bash
# Start one server, run the experiment suite against it, stop it.
#   scripts/bench.sh <llamacpp|mlx|lmstudio-gguf|lmstudio-mlx> [experiment ...]
#   scripts/bench.sh fake ceiling
#   ONLY="r8" scripts/bench.sh lmstudio-mlx rate   # re-run single points
#   REPS="2 3" scripts/bench.sh llamacpp concurrency  # repeats: <server>_<x>_rep<N>.json
#   LLAMA_ARGS="--cache-ram 0" VARIANT=cache-ram0 scripts/bench.sh llamacpp prefix
#     # llama-server flags; results labelled llamacpp-cache-ram0, next to the default runs
#   SIZE=4B scripts/bench.sh mlx concurrency prompt  # Qwen3-4B; results labelled mlx-4b
# Experiments: concurrency rate arrivals prompt prefix (default: all), chat.
# Every run waits until the 1-minute load average is below MAX_LOAD and other apps'
# GPU use is below MAX_GPU.
set -euo pipefail
cd "$(dirname "$0")/.."

MODELS=${MODELS:-$HOME/.lmstudio/models/lmstudio-community}
SIZE=${SIZE:-1.7B}  # Qwen3 size, as in the lmstudio-community repo names
GGUF=$MODELS/Qwen3-$SIZE-GGUF/Qwen3-$SIZE-Q4_K_M.gguf
MLXDIR=$MODELS/Qwen3-$SIZE-MLX-4bit
size=$(echo $SIZE | tr A-Z a-z)
LMS=${LMS:-$HOME/.lmstudio/bin/lms}
SLOTS=8       # concurrent sequences every server is allowed to batch
CTX=32768     # total KV budget (llama.cpp splits it across slots: 4096 each)
MAX_LOAD=${MAX_LOAD:-6}
MAX_GPU=${MAX_GPU:-20}  # % busy from other apps (load average does not see GPU work)
LOGS=.logs; mkdir -p $LOGS
RESULTS=${RESULTS:-results}  # point dry runs on a busy machine elsewhere
REPS=${REPS:-1}  # repeat numbers; rep 1 keeps the plain file name

server=$1; shift
[ $SIZE = 1.7B ] || VARIANT=$size${VARIANT:+-$VARIANT}
label=$server${VARIANT:+-$VARIANT}
exps=${*:-concurrency rate arrivals prompt prefix}

# Qwen3 thinks before answering unless the chat template is told not to; with thinking on,
# TTFT would time the first reasoning token. llama.cpp and LM Studio take it per request.
NO_THINK='{"chat_template_kwargs": {"enable_thinking": false}}'

case $server in
llamacpp)
  URL=http://127.0.0.1:8081 MODEL=qwen3-$size EOS=--ignore-eos
  start() { llama-server -m "$GGUF" --port 8081 -np $SLOTS -c $CTX --log-disable ${LLAMA_ARGS:-} \
              >$LOGS/$server.log 2>&1 & }
  stop() { pkill -f "llama-server -m $GGUF --port 8081" || true; }
  version() { llama-server --version 2>&1 | grep version; } ;;
mlx)
  # mlx_lm.server has no ignore_eos; random-word prompts make early EOS rare and
  # mean_output_tokens in each result shows whether it happened.
  URL=http://127.0.0.1:8082 MODEL=default_model EOS= NO_THINK='{}'  # set on the server instead
  start() { uv run --group mlx mlx_lm.server --model "$MLXDIR" --port 8082 --decode-concurrency $SLOTS \
              --prompt-concurrency $SLOTS --chat-template-args '{"enable_thinking": false}' \
              --log-level WARNING >$LOGS/$server.log 2>&1 & }
  stop() { pkill -f "mlx_lm.server --model $MLXDIR --port 8082" || true; }
  version() { echo "mlx-lm $(uv run --group mlx python -c 'import mlx_lm, mlx.core as m; print(mlx_lm.__version__, "mlx", m.__version__)')"; } ;;
lmstudio-gguf|lmstudio-mlx)
  URL=http://127.0.0.1:1234 MODEL=bench EOS=--ignore-eos
  key=qwen3-$size; [ $server = lmstudio-mlx ] && key=qwen3-$size-mlx
  # Other jobs may use LM Studio too: only unload our own model instance, and only stop
  # the HTTP server if this script started it.
  OWN_SERVER=0; $LMS server status 2>&1 | grep -q ON || OWN_SERVER=1
  start() { $LMS server start --port 1234 >/dev/null
            $LMS load $key --parallel $SLOTS -c $CTX --identifier bench -y >$LOGS/$server.log 2>&1; }
  stop() { $LMS unload bench >/dev/null 2>&1 || true
           [ $OWN_SERVER = 1 ] && { $LMS server stop >/dev/null 2>&1 || true; }; }
  version() { echo "LM Studio $($LMS version 2>/dev/null | grep -o 'CLI commit.*' || true);" \
                   "$($LMS runtime ls 2>/dev/null | grep '✓' | awk '{print $1}' | tr '\n' ' ')"; } ;;
fake)  # measures the client's own ceiling: 1 ms/token streams, more slots than users
  URL=http://127.0.0.1:8090 MODEL=fake EOS=
  start() { uv run python -m servebench.fakeserver --port 8090 --slots 64 --decode-s 0.001 \
              --prefill-s-per-token 0 >$LOGS/$server.log 2>&1 & }
  stop() { pkill -f "servebench.fakeserver --port 8090" || true; }
  version() { echo "fake 1ms/token"; } ;;
*) echo "unknown server $server" >&2; exit 2 ;;
esac

gpu() {  # GPU utilization %, from the same IOAccelerator counter results record
  ioreg -r -d 1 -c IOAccelerator | grep -o '"Device Utilization %"=[0-9]*' | head -1 | cut -d= -f2
}

wait_calm() {  # our server is idle here, so GPU use is someone else's
  while true; do
    load=$(sysctl -n vm.loadavg | awk '{print $2}') g=$(gpu)
    awk "BEGIN{exit !($load < $MAX_LOAD && ${g:-0} < $MAX_GPU)}" && return
    echo "$(date +%H:%M) load $load (max $MAX_LOAD), gpu ${g}% (max $MAX_GPU), waiting..." >&2
    sleep 30
  done
}

bench() {  # bench <experiment> <name> [servebench run args...]
  local exp=$1 name=$2; shift 2
  [[ -z "${ONLY:-}" || " $ONLY " == *" $name "* ]] || return 0  # ONLY="r8 c4": re-run single points
  local suffix=; [ $REP = 1 ] || suffix=_rep$REP
  wait_calm
  uv run servebench run --url $URL --model $MODEL --label $label $EOS \
    --out $RESULTS/$exp/${label}_$name$suffix.json --tag experiment=$exp --tag "engine=$VERSION" \
    --tag rep=$REP --tag "llama_args=${LLAMA_ARGS:-}" "$@"
  sleep 5
}

trap stop EXIT
stop; start
for _ in $(seq 120); do curl -sf $URL/v1/models >/dev/null && break; sleep 1; done
VERSION=$(version)
echo "$label: $VERSION ${LLAMA_ARGS:-}"

# Repeats run as whole passes, so the spread includes drift over time, not only run-to-run noise.
for REP in $REPS; do
for exp in $exps; do case $exp in
concurrency)  # closed loop, default 256-word (~268-token) prompts, 128 output tokens
  for c in 1 2 4 8 16 32; do
    bench concurrency c$c --concurrency $c --n $((c < 4 ? 16 : 4 * c)) --tag x=$c
  done ;;
rate)  # open loop Poisson arrivals, ~40 s per offered rate
  for r in 1 2 3 4 6 8; do
    bench rate r$r --pattern poisson --rate $r --n $(awk "BEGIN{print int($r*40)}") --tag x=$r
  done ;;
arrivals)  # same mean rate, different burstiness
  for p in constant poisson burst; do
    bench arrivals $p --pattern $p --rate 2 --burst 8 --n 64 --tag x=$p
  done ;;
prompt)  # single user: prefill cost vs prompt length
  for t in 128 512 1024 2048 3584; do
    bench prompt t$t --concurrency 1 --n 8 --prompt-tokens $t --max-tokens 64 --tag x=$t
  done ;;
prefix)  # 2048-token prompts, 4 users, growing shared prefix
  for s in 0 1024 1920; do
    bench prefix s$s --concurrency 4 --n 32 --prompt-tokens 2048 --shared-prefix $s \
      --max-tokens 64 --tag x=$s
  done ;;
chat)  # chat template, lognormal lengths around 256 prompt / 128 output tokens
  for c in 1 8; do
    bench chat c$c --endpoint chat --concurrency $c --n $((c < 4 ? 16 : 4 * c)) \
      --prompt-sigma 0.6 --output-sigma 0.6 --extra-body "$NO_THINK" --tag x=$c
  done ;;
ceiling)  # run with the fake server: how fast can the client ingest tokens?
  for c in 1 8 32 64; do
    bench ceiling c$c --concurrency $c --n $((2 * c)) --max-tokens 512 --tag x=$c
  done ;;
*) echo "unknown experiment $exp" >&2; exit 2 ;;
esac; done
done
