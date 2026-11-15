PYTHON ?= python
MEMCAP ?=

.PHONY: install install-locked data run train serve test lint clean

install:
	$(PYTHON) -m pip install -e ".[dev]"

# Exact package versions that produced the committed reports/ (see requirements.lock).
install-locked:
	$(PYTHON) -m pip install -r requirements.lock
	$(PYTHON) -m pip install -e . --no-deps

# Download German Credit (Statlog) from OpenML into data/ (cached after the first call).
data:
	$(MEMCAP) $(PYTHON) -c "from credit_risk.data import load_german_credit; load_german_credit()"

# Full reproduction: 5x5 repeated CV of all models, calibration, cost analysis, fairness audit,
# figures, scorecard tables -> reports/, and the served model -> models/scorecard.joblib.
run: data
	$(MEMCAP) $(PYTHON) -m credit_risk.cli run

# Fit only the served scorecard (what the Docker build runs).
train: data
	$(MEMCAP) $(PYTHON) -m credit_risk.cli train

serve:
	$(PYTHON) -m credit_risk.cli serve

test:
	$(PYTHON) -m pytest -q

lint:
	$(PYTHON) -m ruff check .

clean:
	rm -rf .pytest_cache .ruff_cache build dist src/*.egg-info
