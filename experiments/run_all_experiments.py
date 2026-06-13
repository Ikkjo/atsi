"""Run all 6 experiments (3 scenarios x 2 mic configurations) sequentially.

Usage:
    .venv/bin/python experiments/run_all_experiments.py

    .venv/bin/python experiments/run_all_experiments.py --scenario scenario3

    .venv/bin/python experiments/run_all_experiments.py --mic ihm

    .venv/bin/python experiments/run_all_experiments.py --fail-fast

    .venv/bin/python experiments/run_all_experiments.py --no-cache-asr --no-cache-embeddings
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.run_experiment import run_experiment


def discover_configs(config_dir: Path, scenario: str | None = None, mic: str | None = None) -> list[Path]:
    """Discover experiment config files, optionally filtered."""
    configs = sorted(config_dir.glob("*.json"))
    if scenario:
        configs = [c for c in configs if scenario in c.stem]
    if mic:
        configs = [c for c in configs if mic in c.stem]
    return configs


def run_all_experiments(
    config_dir: Path,
    scenario: str | None = None,
    mic: str | None = None,
    fail_fast: bool = False,
    output_dir: Path | None = None,
    use_cached_asr: bool | None = None,
    use_cached_embeddings: bool | None = None,
) -> dict:
    """Run all matching experiments and aggregate results.

    Returns:
        Dict with all_runs list and summary.
    """
    configs = discover_configs(config_dir, scenario=scenario, mic=mic)
    if not configs:
        print(f"No config files found in {config_dir} matching scenario={scenario} mic={mic}")
        return {"all_runs": [], "summary": {"status": "no_configs"}}

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    all_runs = []
    failed_configs = []

    for config_path in configs:
        print(f"\n{'='*60}")
        print(f"Running: {config_path.name}")
        print(f"{'='*60}")
        try:
            result = run_experiment(
                config_path=config_path,
                output_dir=output_dir,
                fail_fast=fail_fast,
                use_cached_asr=use_cached_asr,
                use_cached_embeddings=use_cached_embeddings,
            )
            all_runs.append({
                "config_path": str(config_path),
                "output_dir": result["output_dir"],
                "experiment_id": result["experiment_id"],
                "status": "completed",
                "summary_path": str(Path(result["output_dir"]) / "metrics" / "summary.json"),
                "summary": result.get("summary", {}),
            })
        except Exception as exc:
            print(f"FAILED: {config_path.name}: {exc}", file=sys.stderr)
            all_runs.append({
                "config_path": str(config_path),
                "output_dir": None,
                "experiment_id": None,
                "status": "failed",
                "error": str(exc),
                "summary_path": None,
                "summary": {},
            })
            failed_configs.append(str(config_path))
            if fail_fast:
                print("Fail-fast enabled. Aborting remaining experiments.", file=sys.stderr)
                break

    # Save all_runs summary
    all_runs_path = Path("results/experiments") / f"all_runs_{timestamp}.json"
    all_runs_path.parent.mkdir(parents=True, exist_ok=True)
    with open(all_runs_path, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": timestamp,
            "num_configs": len(configs),
            "num_completed": len([r for r in all_runs if r["status"] == "completed"]),
            "num_failed": len([r for r in all_runs if r["status"] == "failed"]),
            "all_runs": all_runs,
        }, f, indent=2, default=str)
    print(f"\nAll-runs summary saved: {all_runs_path}")

    if failed_configs:
        print(f"\nWARNING: {len(failed_configs)} experiments failed:")
        for fc in failed_configs:
            print(f"  - {fc}")

    return {
        "all_runs_path": str(all_runs_path),
        "all_runs": all_runs,
        "num_failed": len(failed_configs),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir", type=Path, default=PROJECT_ROOT / "experiments" / "configs", help="Directory containing config JSON files")
    parser.add_argument("--scenario", type=str, help="Filter by scenario (e.g. scenario1)")
    parser.add_argument("--mic", type=str, help="Filter by microphone (e.g. ihm)")
    parser.add_argument("--fail-fast", action="store_true", help="Stop on first failed experiment")
    parser.add_argument("--output-dir", type=Path, help="Base output directory for all runs")
    parser.add_argument("--no-cache-asr", action="store_true", help="Ignore cached ASR and re-run Whisper for all experiments")
    parser.add_argument("--no-cache-embeddings", action="store_true", help="Ignore cached embeddings and re-run ECAPA for all experiments")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run_all_experiments(
        config_dir=args.config_dir,
        scenario=args.scenario,
        mic=args.mic,
        fail_fast=args.fail_fast,
        output_dir=args.output_dir,
        use_cached_asr=not args.no_cache_asr if args.no_cache_asr else None,
        use_cached_embeddings=not args.no_cache_embeddings if args.no_cache_embeddings else None,
    )
    return 0 if result["num_failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
