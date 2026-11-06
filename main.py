import argparse
from dataclasses import fields

from config import ExperimentConfig
from experiment import ATTACKS, STEP_SCHEDULES, run_experiment
import defenses
import predefenses


DEFAULTS = {f.name: f for f in fields(ExperimentConfig)}


def parse_args() -> ExperimentConfig:
    parser = argparse.ArgumentParser(
        description="Federated Learning Experiment Configuration"
    )

    def add(short, name, help=None, choices=None):
        field = DEFAULTS[name]
        shown = f"{help} (default: {field.default})" if help else f"default: {field.default}"
        parser.add_argument(short, '--' + name, default=field.default, type=field.type, choices=choices, help=shown)

    # --- FL Setup ---
    add('-dn', 'dataname', choices=sorted(STEP_SCHEDULES))
    add('-dp', 'partition', choices=['iid', 'noniiddir'])
    add('-n', 'users_count', help='number of participating users')
    add('-f', 'mal_prop', help='actual proportion of malicious users')
    add('-est_f', 'est_mal_prop', help='proportion of malicious users assumed by the defenses')
    add('-pd', 'predefense', choices=sorted(predefenses.predefend))
    add('-d', 'defense', choices=sorted(defenses.defend))
    add('-a', 'attack', choices=sorted(ATTACKS))
    add('-b', 'batch_size', help='batch_size')
    add('-r', 'rounds')
    add('-l', 'learning_rate', help='learning rate')
    add('-m', 'momentum', help='local momentum beta')
    add('-e', 'local_iteration', help='local iteration rounds')
    add('-ts', 'test_step')
    add('-s', 'randomseed')

    # --- Model and schedule (the model itself follows from --dataname) ---
    add('-is', 'img_size', help='DeiT input resolution, CIFAR100 only (patch size 16)')
    add('-ls', 'lr_schedule', choices=['step', 'cosine'])
    add('-wr', 'warmup_rounds', help='warmup rounds for the cosine schedule')
    add('-be', 'beta', help='Dirichlet concentration for noniiddir')

    args = parser.parse_args()

    return ExperimentConfig(**vars(args))


if __name__ == "__main__":
    config = parse_args()
    run_experiment(config)
