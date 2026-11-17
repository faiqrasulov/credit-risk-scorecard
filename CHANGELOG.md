# Changelog

Notable changes to this project, newest first.

## 0.1.0 (2026-11-17)

First tagged release. It contains everything below.

## 2026-11-16

- Package versions pinned in `requirements.lock`, with a `make install-locked` target.
- README: the boosting grid was widened once after a first run; fairness intervals hold the model fixed.

## 2026-11-15

- First version: WoE/IV scorecard against gradient boosting, cost-optimal cutoffs, calibration, reason codes, fairness audit, FastAPI service.
- README: expected cost reported as a tie between the two models, not a win.
