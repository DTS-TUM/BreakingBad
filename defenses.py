import torch
from math import floor


class DefenseTypes:
    NoDefense = 'NoDefense'
    HonestOracle = 'HonestOracle'
    Krum = 'Krum'
    TrimmedMean = 'TrimmedMean'
    MeaMed = 'MeaMed'
    MultiKrum = 'MultiKrum'
    Median = 'Median'
    GeoMed = 'GeoMed'
    Prodigy = 'Prodigy'
    CClip = 'CClip'
    Gatscha = 'Gatscha'
    GatschaAgn = 'GatschaAgn'
    GatschaAvg = 'GatschaAvg'

    def __str__(self):
        return self.value


def _pairwise_sq_distances(clients_grads):
    d = torch.cdist(clients_grads, clients_grads, compute_mode='donot_use_mm_for_euclid_dist')
    d = d * d
    return d


def no_defense(clients_grads, clients_count, corrupted_count, center, honest_mask):
    return clients_grads.mean(dim=0)


def honest_oracle(clients_grads, clients_count, corrupted_count, center, honest_mask):

    if honest_mask.numel() != clients_grads.shape[0]:
        raise ValueError(f"honest_oracle got {clients_grads.shape[0]} gradients but {honest_mask.numel()} mask entries; "
                         "it cannot follow a pre-defense that changes the number of rows")
    return clients_grads[honest_mask].mean(dim=0)


def median(clients_grads, clients_count, corrupted_count, center, honest_mask):
    return clients_grads.median(dim=0).values


def RFA(clients_grads, clients_count, corrupted_count, center, honest_mask):
    m = clients_count
    nu = 0.1
    T = 3
    alpha = 1.0 / m
    z = torch.zeros_like(clients_grads[0])
    for _ in range(T):
        distances = torch.linalg.norm(z - clients_grads, dim=1)         
        betas = alpha / torch.clamp(distances, min=nu) 
        z = (betas.unsqueeze(1) * clients_grads).sum(dim=0) / betas.sum()
    return z


CCLIP_TAU = 10.0
CCLIP_ITERS = 3
def centeredclipping(clients_grads, clients_count, corrupted_count, center, honest_mask):
    """Centered clipping (Karimireddy, He and Jaggi, 2021)."""
    v = center.clone()
    for _ in range(CCLIP_ITERS):
        diff = clients_grads - v
        norms = torch.linalg.norm(diff, dim=1, keepdim=True)
        factor = torch.clamp(CCLIP_TAU / norms.clamp_min(1e-12), max=1.0)
        v = v + (diff * factor).mean(dim=0)
    return v


def krum(clients_grads, clients_count, corrupted_count, center, honest_mask, distances=None, return_index=False, debug=False):
    if not return_index:
        assert clients_count >= 2 * corrupted_count + 1, ('clients_count>=2*corrupted_count + 3', clients_count, corrupted_count)
    neighborhood = clients_count - corrupted_count - 2

    D = _pairwise_sq_distances(clients_grads)
    sorted_D, _ = torch.sort(D, dim=1)
    scores = sorted_D[:, 1:neighborhood+1].sum(dim=1)
    minimal_error_index = int(torch.argmin(scores).item())

    if return_index:
        return minimal_error_index
    return clients_grads[minimal_error_index]


def multi_krum(clients_grads, clients_count, corrupted_count, center, honest_mask, distances=None, debug=False):
    neighborhood = clients_count - corrupted_count - 2
    selected_count = clients_count - corrupted_count

    D = _pairwise_sq_distances(clients_grads)
    sorted_D, _ = torch.sort(D, dim=1)
    scores = sorted_D[:, 1:neighborhood+1].sum(dim=1)
    idx_to_be_averaged = torch.argsort(scores)[:selected_count]
    return clients_grads[idx_to_be_averaged].mean(dim=0)


def meanofmedian(clients_grads, clients_count, corrupted_count, center, honest_mask):
    number_to_consider = clients_count - 2 * corrupted_count
    med = clients_grads.median(dim=0).values
    diff = clients_grads - med
    indices = torch.argsort(diff.abs(), dim=0)
    sorted_diff = torch.gather(diff, 0, indices)
    mean_values = sorted_diff[:number_to_consider].mean(dim=0)
    return mean_values + med

def t_mean(clients_grads, clients_count, corrupted_count, center=None, honest_mask=None):
    sorted_grads, _ = torch.sort(clients_grads, dim=0)
    trimmed = sorted_grads[corrupted_count : clients_count - corrupted_count]
    return trimmed.mean(dim=0)   


def _neighbor_stats(clients_grads):
    n = clients_grads.shape[0]
    D = _pairwise_sq_distances(clients_grads)
    sorted_vals, sorted_idx = torch.sort(D, dim=1)      
    return sorted_vals, sorted_idx


def _weighted_average(clients_grads, score):
    sumscore = score.sum()
    return ((score.unsqueeze(1) / sumscore) * clients_grads).sum(dim=0)


def prodigy(clients_grads, clients_count, corrupted_count, center, honest_mask):
    n = clients_count
    neighborhood = corrupted_count 
    sorted_vals, sorted_idx = _neighbor_stats(clients_grads)

    score1 = torch.empty(n, device=clients_grads.device, dtype=clients_grads.dtype)
    for client in range(n):
        grp = clients_grads[sorted_idx[client, 0:neighborhood]]
        mean_grads = grp.mean(dim=0)
        std_dev_grads = grp.var(dim=0, unbiased=False).sqrt()
        score1[client] = torch.linalg.norm(std_dev_grads) / torch.linalg.norm(mean_grads)

    score2 = sorted_vals[:, neighborhood: n - neighborhood].sum(dim=1)

    score = score1 / score2
    zero_idx = torch.argsort(score)[:corrupted_count]
    score[zero_idx] = 0
    return _weighted_average(clients_grads, score)

    
def gatscha(clients_grads, clients_count, corrupted_count, center, honest_mask):
    sorted_vals, sorted_idx = _neighbor_stats(clients_grads)

    score1 = torch.empty(clients_count, device=clients_grads.device, dtype=clients_grads.dtype)
    for client in range(clients_count):
        grp = clients_grads[sorted_idx[client, 0:corrupted_count]]
        std_dev_grads = grp.var(dim=0, unbiased=False).sqrt()
        score1[client] = torch.linalg.norm(std_dev_grads)

    score2 = 1.0 / sorted_vals[:, clients_count - corrupted_count - 1]   

    score = score1 * score2
    lowest_keys = torch.argsort(score2)[:corrupted_count]              
    score[lowest_keys] = 0
    return _weighted_average(clients_grads, score)


def gatscha_agn(clients_grads, clients_count, corrupted_count, center, honest_mask):
    sorted_vals, _ = _neighbor_stats(clients_grads)
    f_conservative = floor((clients_count-1)/2)
    score1 = sorted_vals[:, 1]

    score2 = 1.0 / sorted_vals[:, clients_count - f_conservative - 1]   

    score = score1 * score2
    lowest_keys = torch.argsort(score2)[:f_conservative]              
    score[lowest_keys] = 0
    return _weighted_average(clients_grads, score)
    
    
def gatscha_avg(clients_grads, clients_count, corrupted_count, center, honest_mask):
    sorted_vals, _ = _neighbor_stats(clients_grads)

    score2 = 1.0 / sorted_vals[:, clients_count - corrupted_count - 1]   
    lowest_keys = torch.argsort(score2)[:corrupted_count]    
    score = torch.full((clients_count,), 1.0 / (clients_count - corrupted_count),
                   device=score2.device)
    score[lowest_keys] = 0
    return _weighted_average(clients_grads, score)

defend = {DefenseTypes.Krum: krum,
          DefenseTypes.MultiKrum: multi_krum,
          DefenseTypes.Median: median,
          DefenseTypes.MeaMed: meanofmedian,
          DefenseTypes.TrimmedMean: t_mean,
          DefenseTypes.NoDefense: no_defense,
          DefenseTypes.HonestOracle: honest_oracle,
          DefenseTypes.Prodigy: prodigy,
          DefenseTypes.GeoMed: RFA,
          DefenseTypes.CClip: centeredclipping,
          DefenseTypes.Gatscha: gatscha,
          DefenseTypes.GatschaAgn: gatscha_agn,
          DefenseTypes.GatschaAvg: gatscha_avg,
          }