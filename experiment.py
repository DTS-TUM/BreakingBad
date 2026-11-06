import csv
import math
import os
import random
import time
from statistics import NormalDist
from config import ExperimentConfig
import torch
import numpy as np


import client
import datasets
import models
import server
import malicious

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')

def byzantine_count(proportion, users_count):
    return int(round(proportion * users_count))

def alie_z(users_count, fhat):
    s = math.ceil((users_count + 1) / 2) - fhat
    p = (users_count - s) / users_count
    if not 0.5 < p < 1:
        raise ValueError(f"ALIE has no positive finite z for f={fhat} Byzantine clients out of n={users_count}")
    return NormalDist().inv_cdf(p)

def set_global_determinism(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _searching(attack_class, *args):
    return lambda cfg, fhat: attack_class(*args, cfg.predefense, cfg.defense, fhat)


ATTACKS = {
    'NoAttack': lambda cfg, fhat: malicious.NoAttack(),
    'ALIE': lambda cfg, fhat: malicious.ALIE(alie_z(cfg.users_count, fhat)),
    'ALIE_optim': lambda cfg, fhat: malicious.ALIE_optim(alie_z(cfg.users_count, fhat), cfg.predefense, cfg.defense, fhat),
    'FOE': lambda cfg, fhat: malicious.FOE(0.1),
    'FOE_optim': _searching(malicious.FOE_optim, 0.1),
    'FOE_optim1': _searching(malicious.FOE_optim, 1),
    'FOE_optim10': _searching(malicious.FOE_optim, 10),
    'FOE_optim100': _searching(malicious.FOE_optim, 100),
    'FOE_optim_log': _searching(malicious.FOE_optim_log),
    'SF': lambda cfg, fhat: malicious.SF(),
    'LF': lambda cfg, fhat: malicious.LF(),
    'ZeroMean': lambda cfg, fhat: malicious.simple_nomean(),
    'Mimic': lambda cfg, fhat: malicious.MimicVariant(warmup=cfg.rounds),
    'MinMax': _searching(malicious.MinMax),
    'MinSum': _searching(malicious.MinSum),
}


def set_attack(configuration: ExperimentConfig):
    if configuration.attack not in ATTACKS:
        raise ValueError(f"Unknown attack: {configuration.attack}")
    fhat = byzantine_count(configuration.est_mal_prop, configuration.users_count)
    return ATTACKS[configuration.attack](configuration, fhat)

STEP_SCHEDULES = {
    'CIFAR10': [(1, 8, 0.2),
                (2, 4, 0.5),
                (3, 4, 0.2)],  
    'FEMNIST': [(2, 3, 0.1)],
    'Shakespeare': [(7, 10, 0.1)],
    'CIFAR100': [(5, 8, 0.2), (7, 8, 0.5)],
}


def learning_rate_at(config: ExperimentConfig, round):
    base = config.learning_rate
    if config.lr_schedule != 'cosine':
        for num, den, factor in STEP_SCHEDULES[config.dataname]:
            if round >= num * config.rounds // den:
                base *= factor
        return base
    W = config.warmup_rounds
    if round < W:
        return base * (round + 1) / W
    progress = (round - W) / max(1, config.rounds - W)
    return base * 0.5 * (1.0 + math.cos(math.pi * progress))


def result_filename(config: ExperimentConfig):
    name = '{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(config.dataname, config.partition, config.users_count, config.mal_prop, config.est_mal_prop, config.predefense, config.defense, config.attack, config.batch_size, config.rounds, config.learning_rate, config.momentum, config.local_iteration, config.test_step, config.randomseed)
    extra = []
    if models.uses_deit(config.dataname):
        extra.append('DeiT{}'.format(config.img_size))
    if config.lr_schedule != 'step':
        extra.append('{}w{}'.format(config.lr_schedule, config.warmup_rounds))
    if config.partition == 'noniiddir' and config.beta != 0.5:
        extra.append('beta{}'.format(config.beta))
    if extra:
        name += '_' + '_'.join(extra)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    return os.path.join(RESULTS_DIR, name + '.txt')


def format_duration(seconds):
    seconds = int(round(seconds))
    return '{:d}h{:02d}m{:02d}s'.format(seconds // 3600, seconds % 3600 // 60, seconds % 60)


def _wait_for_device(device):
    if device.type == 'cuda':
        torch.cuda.synchronize()
    elif device.type == 'mps':
        torch.mps.synchronize()


def record_timing(config: ExperimentConfig, timing):
    path = os.path.join(RESULTS_DIR, 'timings.csv')
    row = {'finished': time.strftime('%Y-%m-%d %H:%M:%S'),
           'result_file': os.path.basename(result_filename(config)),
           **{k: (round(v, 2) if isinstance(v, float) else v) for k, v in timing.items()}}
    is_new = not os.path.isfile(path)
    with open(path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def generate_distribution_map(config: ExperimentConfig):
    beta = config.beta
    distribution_map = datasets.find_distribution_map(config.dataname, config.partition, config.users_count, config.batch_size, beta)
    return distribution_map

def generate_clients(config: ExperimentConfig, distribution_map, attacker, device):

    clients = []
    for client_id in range(config.users_count):
        if client_id < byzantine_count(config.mal_prop, config.users_count):
            is_mal = True
        else:
            is_mal = False
        clients.append(client.Client(client_id, config.local_iteration, config.batch_size, is_mal, config.momentum, config.dataname, distribution_map[client_id], device, img_size=config.img_size))

    mal_clients = [u for u in clients if u.is_malicious]

    if config.attack == 'LF':
        attacker.attack(mal_clients, clients)

    return clients, mal_clients

def evaluate(config: ExperimentConfig, the_server, round):
    print('testing starts...')
    hist = None
    if config.dataname == 'Shakespeare':
        test_loss, correct, perplexity, total_chars = the_server.test()
        accuracy = 100. * float(correct) / total_chars
        print('Test set: [{:3d}] Learning rate: {:.4f}, Average loss: {:.4f}, Perplexity: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(round, the_server.learning_rate, test_loss, perplexity, correct, total_chars, accuracy))
        hist = (round, the_server.learning_rate, perplexity, accuracy, test_loss)
    else:
        test_loss, correct, total_chars = the_server.test()
        accuracy = 100. * float(correct) / total_chars
        print('Test set: [{:3d}] Learning rate: {:.4f}, Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(round, the_server.learning_rate, test_loss,
                                                                                        correct,
                                                                                        total_chars,
                                                                        accuracy))
        hist = (round, the_server.learning_rate, accuracy, test_loss)

    return hist, test_loss

def run_experiment(config: ExperimentConfig):
    print("Running experiment with config:")
    print(config)
    start_time = time.perf_counter()

    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")

    attacker = set_attack(config)

    set_global_determinism(config.randomseed)

    distribution_map = generate_distribution_map(config)
    clients, mal_clients = generate_clients(config, distribution_map, attacker, device)
    # datasets.print_data_distribution(clients)
    the_server = server.Server(clients, config.est_mal_prop, config.batch_size, config.learning_rate, config.dataname, device, img_size=config.img_size, attacker=attacker)

    n_params = sum(p.numel() for p in the_server.test_net.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")


    TEST_STEP = config.test_step

    setup_seconds = time.perf_counter() - start_time
    train_seconds = 0.0
    test_seconds = 0.0
    train_rounds = 0

    print("\nStarting Testing/Training...")
    accuracies = []
    test_loss = 0
    for round in range(config.rounds+1):
        if round % TEST_STEP == 0 or round == config.rounds: 
            print("test round: ", round)    
            if np.isnan(test_loss):
                print("Test loss is NaN, taking action!")
                hist = (round,) + hist[1:]
                accuracies.append(hist)
                continue

            else:
                phase_start = time.perf_counter()
                hist, test_loss = evaluate(config, the_server, round)
                _wait_for_device(device)
                test_seconds += time.perf_counter() - phase_start
            accuracies.append(hist)
            if train_rounds > 0:
                elapsed = time.perf_counter() - start_time
                remaining = (config.rounds - round) * (train_seconds + test_seconds) / train_rounds
                print('Time: elapsed {}, {:.2f} s per training round, about {} left'.format(
                    format_duration(elapsed), train_seconds / train_rounds, format_duration(remaining)))
            
        
        if round != config.rounds and not np.isnan(test_loss):
            the_server.learning_rate = learning_rate_at(config, round)
                
            print("train round: ", round) 
            phase_start = time.perf_counter()
            the_server.dispatch_weights(round)
               
            if attacker is not None and config.attack != 'LF':
                attacker.attack(mal_clients, clients)
            
            the_server.collect_gradients()
 
            the_server.defend(config.predefense, config.defense, round)   
            _wait_for_device(device)
            train_seconds += time.perf_counter() - phase_start
            train_rounds += 1

    np.savetxt(result_filename(config), accuracies, delimiter=',')

    timing = {'device': str(device), 'train_rounds': train_rounds,
              'setup_seconds': setup_seconds, 'train_seconds': train_seconds, 'test_seconds': test_seconds,
              'total_seconds': time.perf_counter() - start_time,
              'seconds_per_train_round': train_seconds / max(1, train_rounds)}
    record_timing(config, timing)
    print('Wall-clock time: total {} (setup {}, training {}, testing {}), {:.2f} s per training round'.format(
        format_duration(timing['total_seconds']), format_duration(setup_seconds), format_duration(train_seconds),
        format_duration(test_seconds), timing['seconds_per_train_round']))
    return timing
