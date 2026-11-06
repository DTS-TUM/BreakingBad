from dataclasses import dataclass

@dataclass
class ExperimentConfig:
    dataname: str = "CIFAR10"
    partition: str = "iid"
    users_count: int = 51
    mal_prop: float = 0.24        # actual proportion of malicious clients
    est_mal_prop: float = 0.24    # proportion the server (defenses and predefenses) assumes
    predefense: str = "NoDefense"
    defense: str = "NoDefense"
    attack: str = "NoAttack"
    batch_size: int = 64
    rounds: int = 300
    learning_rate: float = 0.1
    momentum: float = 0.0
    local_iteration: int = 1
    test_step: int = 10
    randomseed: int = 0
    img_size: int = 112           # DeiT input resolution (CIFAR100 only); images are upsampled to this on the GPU
    lr_schedule: str = "step"     # "step" or "cosine" (linear warmup, then cosine)
    warmup_rounds: int = 50        # used by the cosine schedule
    beta: float = 0.5             # Dirichlet concentration for partition='noniiddir' (was hard coded to 0.5)