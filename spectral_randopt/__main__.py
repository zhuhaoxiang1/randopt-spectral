from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description='Spectral RandOpt: offline smoke and Qwen 1.5B experiments')
    commands = parser.add_subparsers(dest='command', required=True)
    smoke_parser = commands.add_parser('smoke', help='Offline random tiny-Qwen2 end-to-end verification')
    smoke_parser.add_argument('--output-dir', default='runs/smoke')
    smoke_parser.add_argument('--resume', action='store_true')
    run = commands.add_parser('run', help='Run an experiment using already-local weights and data')
    run.add_argument('--config', required=True)
    run.add_argument('--output-dir')
    run.add_argument('--model')
    run.add_argument('--seed', type=int)
    run.add_argument('--device')
    run.add_argument('--dtype', choices=['float32', 'bfloat16', 'float16'])
    run.add_argument('--resume', action='store_true')
    run.add_argument('--search-only', action='store_true', help='Checkpoint search without generating test predictions')
    run.add_argument('--dry-run', action='store_true', help='Validate configuration only; no model, data, or network access')
    analyze_parser = commands.add_parser('analyze', help='Regenerate tables from a completed run')
    analyze_parser.add_argument('--run-dir', required=True)
    budget_parser = commands.add_parser('plan-budget', help='Choose a shared scale from measured timings, never accuracy')
    budget_parser.add_argument('--calibration-dir', required=True)
    budget_parser.add_argument('--config', default='configs/qwen15b-12h.json')
    budget_parser.add_argument('--output-config', required=True)
    budget_parser.add_argument('--hours', type=float, required=True, help='Hours available for the main run, excluding setup')
    budget_parser.add_argument('--safety-factor', type=float, default=1.5)
    prepare_parser = commands.add_parser('prepare', help='Explicitly download model/data; use on the GPU host only')
    prepare_parser.add_argument('--output-dir', default='data/spectral')
    prepare_parser.add_argument('--model', default='Qwen/Qwen2.5-1.5B-Instruct')
    prepare_parser.add_argument('--revision', default='main')
    group = prepare_parser.add_mutually_exclusive_group()
    group.add_argument('--data-only', action='store_true')
    group.add_argument('--model-only', action='store_true')
    args = parser.parse_args()
    if args.command == 'smoke':
        from .smoke import smoke
        print(smoke(args.output_dir, args.resume))
    elif args.command == 'run':
        from .config import load_config
        config = load_config(args.config)
        for key in ('model', 'seed', 'device', 'dtype'):
            if getattr(args, key) is not None:
                setattr(config, key, getattr(args, key))
        if args.output_dir:
            config.output_dir = str(Path(args.output_dir).resolve())
        config.validate()
        if args.dry_run:
            from .runner import make_candidates
            print(json.dumps({'status': 'valid', 'model': config.model, 'methods': len(config.methods),
                              'candidates_per_method': len(make_candidates(config)),
                              'network_access': False, 'model_loaded': False,
                              'selection_samples': config.train_samples, 'probe_samples': config.probe_samples,
                              'test_samples': config.test_samples, 'output_dir': config.output_dir}, indent=2))
        else:
            from .runner import run_experiment
            print(run_experiment(config, resume=args.resume, search_only=args.search_only))
    elif args.command == 'analyze':
        from .analysis import analyze
        print(analyze(args.run_dir))
    elif args.command == 'plan-budget':
        from .budget import plan_budget
        print(json.dumps(plan_budget(args.calibration_dir, args.config, args.output_config,
                                     args.hours, args.safety_factor), ensure_ascii=False, indent=2))
    else:
        from .prepare import prepare
        prepare(args.output_dir, args.model, args.revision, args.data_only, args.model_only)


if __name__ == '__main__':
    main()
