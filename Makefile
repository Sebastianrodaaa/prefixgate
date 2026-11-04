PY ?= python3

.PHONY: test repro all data train score latency baselines run plots banner app demo agent gif

test:            ## checks on real data and the committed scores (no torch)
	$(PY) -m prefixgate.ltt && $(PY) test_prefixgate.py

repro: run plots ## recompute every table and plot from the committed scores

all: data train score latency baselines run plots   ## the whole CPU run, several hours

data:
	$(PY) -m prefixgate.data
train:
	$(PY) -m prefixgate.dm train
score:
	$(PY) -m prefixgate.dm score
latency:
	$(PY) -m prefixgate.dm latency
baselines:
	$(PY) -m prefixgate.dm baselines
run:
	$(PY) -m prefixgate.run
plots:
	$(PY) scripts/plots.py
banner:
	$(PY) scripts/banner.py
app:
	$(PY) app/server.py 8765
agent:          ## Llama 3.2 3B writes 24 tool calls, the gate streams them; results/agent_runs.json, ~6 min
	$(PY) -m prefixgate.agent
demo:           ## the live agent dashboard on :8765 (first live run loads Llama + the gate, ~30 s)
	@echo "open http://127.0.0.1:8765/?agent"
	$(PY) app/server.py 8765
gif:            ## screenshots + GIFs from a running dashboard (needs `make demo` in another shell)
	$(PY) scripts/demo_gif.py
