.PHONY: help data test cov cpp bench backtest sweep tournament clean

help:
	@echo "make data        download and cache the real datasets (~50MB)"
	@echo "make test        run the test suite"
	@echo "make cov         tests with coverage"
	@echo "make cpp         build the C++ Monte Carlo simulator"
	@echo "make bench       C++ determinism + throughput check"
	@echo "make backtest    full walk-forward evaluation on club data"
	@echo "make sweep       strategy configuration sweep with deflated p-values"
	@echo "make tournament  bracket probabilities, exact DP vs Monte Carlo"

data:
	python3 -c "from wcq.data import sources; print(sources.fetch_all())"

test:
	python3 -m pytest tests/

cov:
	python3 -m pytest tests/ --cov=wcq --cov-report=term-missing

cpp:
	$(MAKE) -C cpp

bench: cpp
	@echo "-- determinism across thread counts (checksums must be identical) --"
	@for t in 1 2 4 8; do ./cpp/mc_tournament --sims 2000000 --threads $$t \
	  | grep -E '^ +[0-9]+ ' | cksum | sed "s/\$$/  threads=$$t/"; done
	@echo "-- throughput + exact cross-check --"
	./cpp/mc_tournament --sims 20000000 --check | tail -5

backtest:
	python3 scripts/run_backtest.py --dataset club --devig shin --min-edge 0.03 \
	  --json-out docs/results_club.json

sweep:
	python3 scripts/sweep.py --divisions E0,SP1,D1,I1,F1

tournament: cpp
	python3 scripts/run_tournament.py --as-of 2026-06-11 --sims 2000000

clean:
	$(MAKE) -C cpp clean
	rm -rf .pytest_cache .coverage
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
