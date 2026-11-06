import torch


def no_defense(clients_grads, clients_count, corrupted_count):
    return clients_grads


def nnm(clients_grads, clients_count, corrupted_count):

    neighborhood = clients_count - corrupted_count

    D = torch.cdist(clients_grads, clients_grads, compute_mode='donot_use_mm_for_euclid_dist')
    group_idx = torch.argsort(D, dim=1)[:, :neighborhood]

    resulting_grads = torch.empty_like(clients_grads)
    for client in range(clients_count):
        resulting_grads[client] = clients_grads[group_idx[client]].mean(dim=0)
    return resulting_grads


def bucket(clients_grads, clients_count, corrupted_count):
    bucket_size = 3
    perm = torch.randperm(clients_count, device=clients_grads.device)
    buckets = [perm[i:i + bucket_size] for i in range(0, clients_count, bucket_size)]
    return torch.stack([clients_grads[b].mean(dim=0) for b in buckets])


def arc(clients_grads, clients_count, corrupted_count):
    k = 2 * corrupted_count * (clients_count - corrupted_count) // clients_count  
    norms = torch.linalg.norm(clients_grads, dim=1)

    C = torch.topk(norms, k + 1).values[-1]

    clients_grads = clients_grads.clone()
    mask = norms > C
    if mask.any():
        clients_grads[mask] = clients_grads[mask] * (C / norms[mask]).unsqueeze(1)
    return clients_grads


predefend = {'NoDefense': no_defense, 'NNM': nnm, 'Bucket': bucket, 'ARC': arc}
