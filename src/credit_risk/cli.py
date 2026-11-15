"""Command line: ``credit-risk run | train | serve``."""

from __future__ import annotations

import argparse
import time
from collections.abc import Sequence

from threadpoolctl import threadpool_limits

from credit_risk.data import load_german_credit


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="credit-risk", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name, help_ in (("run", "full reproduction: CV, analyses, figures, served model"),
                        ("train", "fit the served scorecard and its cost-optimal cutoff")):
        p = sub.add_parser(name, help=help_)
        p.add_argument("--data-home", default="data")
        p.add_argument("--model-path", default="models/scorecard.joblib")
        p.add_argument("--folds", type=int, default=5)
        p.add_argument("--repeats", type=int, default=5)
        p.add_argument("--seed", type=int, default=42)
        if name == "run":
            p.add_argument("--reports", default="reports")
    s = sub.add_parser("serve", help="serve the FastAPI app with uvicorn")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--model-path", default=None)
    args = parser.parse_args(argv)

    if args.command == "serve":
        import os

        import uvicorn

        if args.model_path:
            os.environ["CREDIT_RISK_MODEL"] = args.model_path
        uvicorn.run("credit_risk.api:app", host=args.host, port=args.port)
        return

    start = time.perf_counter()
    X, y = load_german_credit(args.data_home)
    # Thousands of tiny fits: OpenMP/BLAS threads only add contention, so use one thread.
    with threadpool_limits(limits=1):
        _run_or_train(args, X, y)
    print(f"done in {time.perf_counter() - start:.1f} s")


def _run_or_train(args: argparse.Namespace, X, y) -> None:
    if args.command == "run":
        from credit_risk.experiment import run_experiment

        m = run_experiment(X, y, reports_dir=args.reports, model_path=args.model_path,
                           n_splits=args.folds, n_repeats=args.repeats, seed=args.seed)
        print(f"wrote {args.reports}/metrics.json; served-model cutoff "
              f"PD >= {m['final_model']['threshold_pd']:.4f}")
    else:
        from credit_risk.train import save_artifact, train_scorecard

        art = train_scorecard(X, y, n_splits=args.folds, n_repeats=args.repeats, seed=args.seed)
        path = save_artifact(art, args.model_path)
        print(f"saved {path}: cutoff PD >= {art['threshold_pd']:.4f} "
              f"(score <= {art['cutoff_score']:.1f}), OOF AUC {art['oof_roc_auc']:.3f}")


if __name__ == "__main__":
    main()
