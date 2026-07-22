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
from pathlib import Path
from typing import Any

import dspy

from react_hover.agent import SafeAgent, build_react_agent
from react_hover.data import load_hover
from react_hover.env import load_env
from react_hover.eval_runner import DEFAULT_STUDENT_LM, run_baseline_eval
from react_hover.history import build_run_record, rows_from_dspy_result, save_run
from react_hover.lm import build_lm
from react_hover.metric import top5_recall
from react_hover.tools import configure_backend

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("react_hover")
load_env()


def _configure_lms(student_lm: str, teacher_lm: str | None, temperature: float) -> dspy.LM:
    student = build_lm(student_lm, temperature=temperature)
    dspy.configure(lm=student)
    return student


def _config_from_args(args: argparse.Namespace, **extra: Any) -> dict[str, Any]:
    cfg = {
        "student_lm": args.student_lm,
        "teacher_lm": getattr(args, "teacher_lm", None),
        "temperature": args.temperature,
        "backend": args.backend,
        "max_iters": args.max_iters,
        "num_threads": args.num_threads,
        "max_errors": args.max_errors,
        "safe": bool(args.safe),
        "train_size": args.train_size,
        "dev_size": args.dev_size,
        "load": getattr(args, "load", None),
    }
    cfg.update(extra)
    return cfg


def _persist_eval(
    *,
    kind: str,
    result,
    config: dict[str, Any],
    parent_id: str | None = None,
    notes: str | None = None,
) -> Path:
    record = build_run_record(
        kind=kind,
        score=float(result.score),
        results=rows_from_dspy_result(result),
        config=config,
        parent_id=parent_id,
        notes=notes,
    )
    path = save_run(record)
    logger.info("Persisted eval run %s → %s", record["id"], path)
    return path


def cmd_demo(args: argparse.Namespace) -> None:
    configure_backend(args.backend)
    _configure_lms(args.student_lm, args.teacher_lm, args.temperature)
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
    configure_backend(args.backend)
    _configure_lms(args.student_lm, args.teacher_lm, args.temperature)

    if not args.teacher_lm:
        raise SystemExit("--teacher-lm is required for optimize (e.g. openai/gpt-4o)")

    teacher = build_lm(args.teacher_lm, temperature=args.temperature)
    trainset, devset, _ = load_hover(
        train_size=args.train_size,
        dev_size=args.dev_size,
    )
    logger.info("Loaded train=%d dev=%d", len(trainset), len(devset))

    react = build_react_agent(max_iters=args.max_iters)
    evaluate = dspy.Evaluate(
        devset=devset,
        metric=top5_recall,
        num_threads=args.num_threads,
        display_progress=True,
        display_table=min(5, len(devset)),
        max_errors=args.max_errors,
    )

    program = SafeAgent(react) if args.safe else react
    logger.info("Running baseline evaluation...")
    baseline = evaluate(program)
    print(f"\nBaseline top5_recall: {baseline.score}")
    base_path = _persist_eval(
        kind="optimize_baseline",
        result=baseline,
        config=_config_from_args(args, auto=args.auto),
        notes="Pre-MIPROv2 baseline",
    )
    base_id = base_path.stem

    # Official tutorial: MIPROv2 with teacher_settings + prompt_model.
    kwargs = dict(
        teacher_settings=dict(lm=teacher),
        prompt_model=teacher,
        max_errors=args.max_errors,
    )
    tp = dspy.MIPROv2(
        metric=top5_recall,
        auto=args.auto,
        num_threads=args.num_threads,
        **kwargs,
    )
    logger.info("Compiling with MIPROv2 (auto=%s)...", args.auto)
    optimized = tp.compile(
        react,
        trainset=trainset,
        max_bootstrapped_demos=args.max_bootstrapped_demos,
        max_labeled_demos=args.max_labeled_demos,
    )

    save_path = Path(args.save or "artifacts/optimized_react.json")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    optimized.save(str(save_path))
    logger.info("Saved optimized program to %s", save_path)

    opt_program = SafeAgent(optimized) if args.safe else optimized
    logger.info("Running post-optimization evaluation...")
    after = evaluate(opt_program)
    print(f"\nOptimized top5_recall: {after.score}")
    print(f"Delta: {after.score - baseline.score:+.2f}")
    after_path = _persist_eval(
        kind="optimize_after",
        result=after,
        config=_config_from_args(
            args,
            auto=args.auto,
            program_path=str(save_path),
            max_bootstrapped_demos=args.max_bootstrapped_demos,
            max_labeled_demos=args.max_labeled_demos,
        ),
        parent_id=base_id,
        notes=f"Post-MIPROv2; delta={after.score - baseline.score:+.2f}",
    )
    print(f"Saved history: {base_path}")
    print(f"Saved history: {after_path}")


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
