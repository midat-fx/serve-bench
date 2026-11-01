test:
	uv run pytest -q

# Full suite on all servers, then SUMMARY.md and plots. Needs the models from the README.
bench:
	for s in llamacpp mlx lmstudio-gguf lmstudio-mlx; do scripts/bench.sh $$s; done
	scripts/bench.sh fake ceiling
	uv run servebench report

report:
	uv run servebench report

.PHONY: test bench report
