"""
Multi-Model LLM Evaluation CLI (local HuggingFace inference)

Quickstart — run one model:
  python -m llm_eval.cli \\
      --dataset data/sample_qa.json \\
      --models Qwen2.5-7B

Run a single model by pointing directly at a snapshot path (no config edit needed):
  python -m llm_eval.cli \\
      --dataset data/sample_qa.json \\
      --model-path /root/autodl-tmp/Qwen2.5-7B-Instruct \\
      --model-name Qwen2.5-7B

Run all models defined in config/models.yaml:
  python -m llm_eval.cli --dataset data/sample_qa.json

Skip BERTScore (faster):
  python -m llm_eval.cli --dataset data/sample_qa.json --no-bertscore

Limit questions for a quick smoke test:
  python -m llm_eval.cli --dataset data/sample_qa.json --limit 5
"""

import argparse
import sys

from .metrics.evaluator import Evaluator
from .models import load_models_from_config
from .models.base import ModelConfig
from .models.hf_local_model import HFLocalModel
from .output.exporter import export_csv, export_summary_csv
from .output.table import render_summary_table
from .runner import EvalRunner


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="llm-eval",
        description="Thesis evaluation CLI: local HuggingFace model inference + BLEU/F1/BERTScore",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--dataset",
        required=True,
        help="Path to QA dataset (.json, .jsonl, or .csv)",
    )

    # --- Model selection (two modes) ---
    source = p.add_argument_group(
        "model source (pick one)",
        "Either use --model-path for a one-off run, or --models to select "
        "from config/models.yaml. If neither is given, all models in the "
        "config are evaluated.",
    )
    source.add_argument(
        "--model-path",
        metavar="PATH",
        help="Absolute path to a local HF snapshot directory. "
             "Use with --model-name to label it in results.",
    )
    source.add_argument(
        "--model-name",
        metavar="NAME",
        default=None,
        help="Display name for the model loaded from --model-path "
             "(default: basename of --model-path).",
    )
    source.add_argument(
        "--models",
        nargs="+",
        metavar="NAME",
        help="Names of models to run from config (space-separated). "
             "Default: all models in config.",
    )
    source.add_argument(
        "--config",
        default="config/models.yaml",
        help="Model configuration YAML (default: config/models.yaml).",
    )

    # --- Generation settings ---
    gen = p.add_argument_group("generation settings")
    gen.add_argument(
        "--max-tokens", type=int, default=512,
        help="Max new tokens per response (default: 512).",
    )
    gen.add_argument(
        "--temperature", type=float, default=0.0,
        help="Sampling temperature; 0 = greedy (default: 0.0).",
    )

    # --- Evaluation settings ---
    ev = p.add_argument_group("evaluation settings")
    ev.add_argument(
        "--no-bertscore",
        action="store_true",
        help="Skip BERTScore computation (much faster).",
    )
    ev.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Evaluate only the first N questions (useful for smoke tests).",
    )

    # --- Output ---
    out = p.add_argument_group("output")
    out.add_argument(
        "--output",
        default="results/eval_results.csv",
        help="Output CSV path (default: results/eval_results.csv).",
    )
    out.add_argument(
        "--no-table",
        action="store_true",
        help="Skip rich summary table; only export CSV.",
    )

    return p


def main() -> None:
    args = build_parser().parse_args()

    # ------------------------------------------------------------------ #
    # Resolve which models to run
    # ------------------------------------------------------------------ #
    if args.model_path:
        # Direct path mode: one model, no config needed
        import os
        name = args.model_name or os.path.basename(args.model_path.rstrip("/\\"))
        config = ModelConfig(
            name=name,
            type="hf_local",
            model_id=name,
            model_path=args.model_path,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )
        models = [HFLocalModel(config)]
    else:
        # Config-file mode
        all_models = load_models_from_config(args.config)
        if args.models:
            requested = set(args.models)
            models = [m for m in all_models if m.name in requested]
            not_found = requested - {m.name for m in models}
            if not_found:
                print(
                    f"WARNING: these model names were not found in config: {not_found}",
                    file=sys.stderr,
                )
            if not models:
                print("ERROR: no matching models found. Aborting.", file=sys.stderr)
                sys.exit(1)
        else:
            models = all_models

    # ------------------------------------------------------------------ #
    # Load dataset
    # ------------------------------------------------------------------ #
    import yaml

    evaluator_kwargs: dict = {}
    if not args.model_path:
        try:
            with open(args.config, encoding="utf-8") as f:
                raw_config = yaml.safe_load(f)
            eval_cfg = raw_config.get("evaluation", {})
            evaluator_kwargs["bertscore_lang"] = eval_cfg.get("bertscore_lang", "en")
            evaluator_kwargs["bertscore_model"] = eval_cfg.get("bertscore_model") or None
        except FileNotFoundError:
            pass

    evaluator = Evaluator(**evaluator_kwargs)
    runner = EvalRunner(models, evaluator)
    dataset = runner.load_dataset(args.dataset)

    if args.limit:
        dataset = dataset[: args.limit]

    use_bertscore = not args.no_bertscore

    print(
        f"\nEvaluating {len(dataset)} question(s) across {len(models)} model(s)"
        + (" [BERTScore enabled]" if use_bertscore else " [BERTScore skipped]")
    )
    print(f"Metrics: BLEU, token-F1, ROUGE{'+ BERTScore' if use_bertscore else ''}\n")

    # ------------------------------------------------------------------ #
    # Run
    # ------------------------------------------------------------------ #
    df = runner.run(dataset, use_bertscore=use_bertscore)

    if not args.no_table:
        render_summary_table(df, use_bertscore=use_bertscore)

    export_csv(df, args.output)
    export_summary_csv(df, args.output)


if __name__ == "__main__":
    main()
