PY ?= python3

.PHONY: test repro all data train score latency baselines run plots banner app demo

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
demo:
	$(PY) scripts/demo_gif.py
