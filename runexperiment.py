import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import itertools
import time
from config import ExperimentConfig
from experiment import format_duration, run_experiment


PRESETS = {
    'CIFAR10': dict(dataname='CIFAR10', partition='noniiddir', beta=0.5, users_count=20,
                    lr_schedule='cosine', warmup_rounds=0,
                    batch_size=64, rounds=5000, momentum=0.9,
                    local_iteration=1, learning_rate=0.05, test_step=10), 
   
    'CIFAR100': dict(dataname='CIFAR100', partition='noniiddir', beta=0.1, users_count=20,
                     lr_schedule='cosine', warmup_rounds=50,
                     batch_size=128, rounds=1000, momentum=0.9,
                     local_iteration=1, learning_rate=0.5, test_step=10),

    'FEMNIST': dict(dataname='FEMNIST', partition='noniiddir', users_count=10,
                    lr_schedule='step',
                    batch_size=64, rounds=2000, momentum=0.9,
                    local_iteration=1, learning_rate=0.05, test_step=10), 

    'Shakespeare': dict(dataname='Shakespeare', partition='noniiddir', users_count=10,
                    lr_schedule='step',
                    batch_size=64, rounds=1000, momentum=0.9,
                    local_iteration=1, learning_rate=10, test_step=10), 

}

DATASETS = ['CIFAR100'] #'CIFAR10','FEMNIST','Shakespeare'
MAL_PROP = [(0.25, 0.25)]
SEEDS = [0]

ATTACKS = [
    'NoAttack',
    # 'ALIE_optim',
    # 'FOE_optim_log',
    # 'LF',
    # 'SF',    
    # 'ZeroMean',
    # 'Mimic',
    # 'MinSum',
    # 'MinMax',
]

DEFENSES = [  
    ('NoDefense', 'NoDefense'),
    # ('NoDefense', 'HonestOracle'),
    # ('NNM', 'Median'),
    # ('NNM', 'MeaMed'),
    # ('NNM', 'TrimmedMean'),
    # ('NNM', 'GeoMed'),
    # ('NNM', 'Krum'),
    # ('NNM', 'CClip'),
    # ('NoDefense', 'Prodigy'),
    # ('NoDefense', 'Gatscha'),

]

EXTRA_CONFIGS = [
    # ExperimentConfig(**PRESETS['CIFAR10'], mal_prop=0.25, est_mal_prop=0.25, predefense='NoDefense', defense='NoDefense', attack='NoAttack', randomseed=0),
]

def build_configs():
    configs = list(EXTRA_CONFIGS)
    for dataname, attack, (predefense, defense), (f,fhat), seed in itertools.product(DATASETS, ATTACKS, DEFENSES, MAL_PROP, SEEDS):
        configs.append(ExperimentConfig(**PRESETS[dataname], mal_prop=f, est_mal_prop=fhat,
                                        predefense=predefense, defense=defense, attack=attack, randomseed=seed))
    return configs


def run_experiments():
    configs = build_configs()
    batch_start = time.perf_counter()
    for i, config in enumerate(configs):
        print(f"\n===== Experiment {i + 1} of {len(configs)} =====")
        run_experiment(config)
        elapsed = time.perf_counter() - batch_start
        remaining = elapsed / (i + 1) * (len(configs) - i - 1)
        print(f"===== {i + 1} of {len(configs)} done, batch elapsed {format_duration(elapsed)}, "
              f"about {format_duration(remaining)} left =====")


if __name__ == "__main__":
    run_experiments()
