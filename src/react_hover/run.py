"""CLI for the official DSPy multi-hop ReAct eval + MIPROv2 optimization example.

Source tutorial: https://dspy.ai/tutorials/agents/

Examples:
  # Single demo claim (smoke test)
  uv run python -m react_hover.run demo --student-lm openai/gpt-5.4-mini

  # Baseline evaluation on a small dev slice
  uv run python -m react_hover.run evaluate --student-lm openai/gpt-5.4-mini --dev-size 5

  # Optimize with MIPROv2 then re-evaluate (costs money / time)
  uv run python -m react_hover.run optimize \\
      --student-lm openai/gpt-5.4-mini --teacher-lm openai/gpt-5.4 \\
      --train-size 20 --dev-size 10 --auto light
"""

from __future__ import annotations

import argparse
import logging
import os

import dspy

from react_hover.agent import build_react_agent
from react_hover.env import load_env
from react_hover.eval_runner import DEFAULT_STUDENT_LM, run_baseline_eval
from react_hover.lm import build_lm
from react_hover.tools import configure_backend

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("react_hover")
load_env()


def cmd_demo(args: argparse.Namespace) -> None:
    configure_backend(args.backend)
    student = build_lm(args.student_lm, temperature=args.temperature)
    dspy.configure(lm=student)
    react = build_react_agent(max_iters=args.max_iters)

    claim = args.claim or "David Gregory was born in 1625."
    logger.info("Running demo claim: %s", claim)
    pred = react(claim=claim)
    titles = getattr(pred, "titles", []) or []
    print("\n=== Demo prediction ===")
    print("claim:", claim)
    print("titles:", titles[:10])
    if getattr(pred, "trajectory", None):
        steps = [k for k in pred.trajectory if k.startswith("tool_name_")]
        print(f"trajectory tool calls: {len(steps)}")
        for key in sorted(pred.trajectory):
            if key.startswith("thought_") or key.startswith("tool_name_"):
                print(f"  {key}: {pred.trajectory[key]}")


def cmd_evaluate(args: argparse.Namespace) -> None:
    out = run_baseline_eval(
        student_lm=args.student_lm,
        backend=args.backend,
        train_size=args.train_size,
        dev_size=args.dev_size,
        max_iters=args.max_iters,
        num_threads=args.num_threads,
        max_errors=args.max_errors,
        safe=args.safe,
        temperature=args.temperature,
        load=args.load,
        notes="CLI evaluate",
    )
    print(f"\nBaseline top5_recall: {out['score']}")
    print(f"Saved history: {out['path']}")


def cmd_optimize(args: argparse.Namespace) -> None:
    if not args.teacher_lm:
        raise SystemExit("--teacher-lm is required for optimize (e.g. openai/gpt-5.4)")

    from react_hover.opt_runner import run_miprov2_optimize

    out = run_miprov2_optimize(
        student_lm=args.student_lm,
        teacher_lm=args.teacher_lm,
        backend=args.backend,
        train_size=args.train_size,
        dev_size=args.dev_size,
        max_iters=args.max_iters,
        num_threads=args.num_threads,
        max_errors=args.max_errors,
        safe=args.safe,
        temperature=args.temperature,
        auto=args.auto,
        max_bootstrapped_demos=args.max_bootstrapped_demos,
        max_labeled_demos=args.max_labeled_demos,
        save=args.save or "artifacts/optimized_react.json",
        notes="CLI optimize",
    )
    print(f"\nBaseline top5_recall: {out['baseline']['score']}")
    print(f"Optimized top5_recall: {out['after']['score']}")
    print(f"Delta: {out['delta']:+.2f}")
    print(f"Program: {out['program_path']}")
    print(f"Saved history: {out['baseline']['path']}")
    print(f"Saved history: {out['after']['path']}")


def _add_shared_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--student-lm",
        default=os.getenv("DSPY_STUDENT_LM", DEFAULT_STUDENT_LM),
        help=f"Student LM (default: {DEFAULT_STUDENT_LM} or $DSPY_STUDENT_LM)",
    )
    p.add_argument(
        "--teacher-lm",
        default=os.getenv("DSPY_TEACHER_LM"),
        help="Teacher LM for MIPROv2 (e.g. openai/gpt-5.4 or $DSPY_TEACHER_LM)",
    )
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument(
        "--backend",
        choices=["auto", "colbert", "wikipedia"],
        default="auto",
        help="Search backend (auto probes ColBERTv2, falls back to Wikipedia)",
    )
    p.add_argument("--max-iters", type=int, default=20, help="ReAct max tool iterations")
    p.add_argument("--num-threads", type=int, default=4)
    p.add_argument("--max-errors", type=int, default=999)
    p.add_argument("--safe", action="store_true", help="Swallow agent exceptions during eval")
    p.add_argument("--train-size", type=int, default=100)
    p.add_argument("--dev-size", type=int, default=100)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    demo = sub.add_parser("demo", help="Run one claim through ReAct")
    _add_shared_args(demo)
    demo.add_argument("--claim", default=None)
    demo.set_defaults(func=cmd_demo)

    ev = sub.add_parser("evaluate", help="Evaluate ReAct with top5_recall")
    _add_shared_args(ev)
    ev.add_argument("--load", default=None, help="Load saved program state JSON")
    ev.set_defaults(func=cmd_evaluate)

    opt = sub.add_parser("optimize", help="MIPROv2 optimize then re-evaluate")
    _add_shared_args(opt)
    opt.add_argument("--auto", default="light", choices=["light", "medium", "heavy"])
    opt.add_argument("--max-bootstrapped-demos", type=int, default=3)
    opt.add_argument("--max-labeled-demos", type=int, default=0)
    opt.add_argument("--save", default="artifacts/optimized_react.json")
    opt.set_defaults(func=cmd_optimize)

    return p


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
