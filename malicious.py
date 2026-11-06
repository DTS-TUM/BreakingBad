import numpy as np 
import torch
import predefenses as predefenses
import defenses as defenses

def _honest_grads(all_users):
    return torch.stack([u.grads for u in all_users if not u.is_malicious])

def _honest_first_mask(honest_count, mal_count, device):
    mask = torch.zeros(honest_count + mal_count, dtype=torch.bool, device=device)
    mask[:honest_count] = True
    return mask


class ALIE(object):
    def __init__(self, num_std):
        self.num_std = num_std
        self.needs_own_grads = num_std == 0
        self.grads_mean = None
        self.grads_stdev = None

    def attack(self, mal_users, all_users):
        if len(mal_users) == 0:
            return

        hon_users_grads = _honest_grads(all_users)
        self.grads_mean = hon_users_grads.mean(dim=0)
        self.grads_stdev = hon_users_grads.var(dim=0, unbiased=False).sqrt()

        if self.num_std == 0:
            return

        mal_grads = self.grads_mean - self.num_std * self.grads_stdev
        for usr in mal_users:
            usr.submit = mal_grads.clone()


class ALIE_optim(object):
    def __init__(self, num_std, predfns, dfns, fhat):
        self.num_std = num_std
        self.needs_own_grads = num_std == 0  
        self.grads_mean = None
        self.grads_stdev = None
        self.predfns = predfns
        self.dfns = dfns
        self.fhat = fhat

    def cal_l2_dist(self, clients_grads, f, center, honest_mask):
        current_grads = predefenses.predefend[self.predfns](clients_grads, len(clients_grads), f)
        current_grads = defenses.defend[self.dfns](current_grads, len(current_grads), f, center, honest_mask)
        return torch.linalg.norm(current_grads - self.grads_mean).item()

    def attack(self, mal_users, all_users):
        if len(mal_users) == 0:
            return

        hon_users_grads = _honest_grads(all_users)
        self.grads_mean = hon_users_grads.mean(dim=0)
        self.grads_stdev = hon_users_grads.var(dim=0, unbiased=False).sqrt()

        if self.num_std == 0:
            return

        z_range = list(np.arange(0.25 * self.num_std, 2.01, 0.25 * self.num_std))
        honest_mask = _honest_first_mask(len(hon_users_grads), len(mal_users), hon_users_grads.device)
        max_dist = 0
        best_z = self.num_std

        for z in z_range:
            mal_grads = self.grads_mean - z * self.grads_stdev
            all_grads = torch.vstack((hon_users_grads, mal_grads.repeat(len(mal_users), 1)))
            dist = self.cal_l2_dist(all_grads, self.fhat, all_users[0].defense_center, honest_mask)
            if dist > max_dist:
                max_dist = dist
                best_z = z
        print("chosen z is:", best_z)
        mal_grads = self.grads_mean - best_z * self.grads_stdev
        for usr in mal_users:
            usr.submit = mal_grads.clone()


class FOE(object):
    needs_own_grads = False   
    def __init__(self, epsilon):
        self.epsilon = epsilon
        self.grads_mean = None

    def attack(self, mal_users, all_users):
        if len(mal_users) == 0:
            return

        hon_users_grads = _honest_grads(all_users)
        self.grads_mean = hon_users_grads.mean(dim=0)

        mal_grads = - self.epsilon * self.grads_mean
        for usr in mal_users:
            usr.submit = mal_grads.clone()


class FOE_optim(object):
    needs_own_grads = False  

    def __init__(self, epsilon, predfns, dfns, fhat):
        self.epsilon = epsilon
        self.grads_mean = None
        self.predfns = predfns
        self.dfns = dfns
        self.fhat = fhat

    def cal_l2_dist(self, clients_grads, f, center, honest_mask):
        current_grads = predefenses.predefend[self.predfns](clients_grads, len(clients_grads), f)
        current_grads = defenses.defend[self.dfns](current_grads, len(current_grads), f, center, honest_mask)
        return torch.linalg.norm(current_grads - self.grads_mean).item()

    def attack(self, mal_users, all_users):
        if len(mal_users) == 0:
            return

        hon_users_grads = _honest_grads(all_users)
        self.grads_mean = hon_users_grads.mean(dim=0)

        eps_range = [0.1 * self.epsilon, 0.2 * self.epsilon, 0.3 * self.epsilon, 0.4 * self.epsilon, 0.5 * self.epsilon,
                     0.6 * self.epsilon, 0.7 * self.epsilon, 0.8 * self.epsilon, 0.9 * self.epsilon, self.epsilon]
        honest_mask = _honest_first_mask(len(hon_users_grads), len(mal_users), hon_users_grads.device)
        max_dist = 0
        best_eps = self.epsilon
        for eps in eps_range:
            mal_grads = - eps * self.grads_mean
            all_grads = torch.vstack((hon_users_grads, mal_grads.repeat(len(mal_users), 1)))
            dist = self.cal_l2_dist(all_grads, self.fhat, all_users[0].defense_center, honest_mask)
            if dist > max_dist:
                max_dist = dist
                best_eps = eps

        print("chosen eps is:", best_eps)
        mal_grads = - best_eps * self.grads_mean
        for usr in mal_users:
            usr.submit = mal_grads.clone()


class FOE_optim_log(object):

    needs_own_grads = False

    def __init__(self, predfns, dfns, fhat, eps_min=0.01, eps_max=100.0, num_points=21):
        self.grads_mean = None
        self.predfns = predfns
        self.dfns = dfns
        self.fhat = fhat
        self.eps_min = eps_min
        self.eps_max = eps_max
        self.num_points = num_points

    def cal_l2_dist(self, clients_grads, f, center, honest_mask):
        current_grads = predefenses.predefend[self.predfns](clients_grads, len(clients_grads), f)
        current_grads = defenses.defend[self.dfns](current_grads, len(current_grads), f, center, honest_mask)
        return torch.linalg.norm(current_grads - self.grads_mean).item()

    def attack(self, mal_users, all_users):
        if len(mal_users) == 0:
            return

        hon_users_grads = _honest_grads(all_users)
        self.grads_mean = hon_users_grads.mean(dim=0)

        eps_range = list(np.logspace(np.log10(self.eps_min), np.log10(self.eps_max), num=self.num_points))
        honest_mask = _honest_first_mask(len(hon_users_grads), len(mal_users), hon_users_grads.device)
        max_dist = 0
        best_eps = eps_range[0]
        for eps in eps_range:
            mal_grads = - eps * self.grads_mean
            all_grads = torch.vstack((hon_users_grads, mal_grads.repeat(len(mal_users), 1)))
            dist = self.cal_l2_dist(all_grads, self.fhat, all_users[0].defense_center, honest_mask)
            if dist > max_dist:
                max_dist = dist
                best_eps = eps

        print("chosen eps is:", best_eps)
        mal_grads = - best_eps * self.grads_mean
        for usr in mal_users:
            usr.submit = mal_grads.clone()


class SF(object):
    needs_own_grads = True 

    def __init__(self):
        pass

    def attack(self, mal_users, all_users):
        if len(mal_users) == 0:
            return
        for usr in mal_users:
            usr.submit = - usr.grads


class LF(object):
    needs_own_grads = True  

    def __init__(self):
        pass

    def attack(self, mal_users, all_users):
        if len(mal_users) == 0:
            return
        for usr in mal_users:
            usr.data_set.fliplabels()


class NoAttack(object):
    needs_own_grads = True

    def __init__(self):
        pass

    def attack(self, mal_users, all_users):
        pass


class simple_nomean(object):
    needs_own_grads = True  

    def __init__(self):
        pass

    def attack(self, mal_users, all_users):
        if len(mal_users) == 0:
            return

        mal_users_grads = torch.stack([usr.grads for usr in mal_users])
        mal_grads_mean = mal_users_grads.mean(dim=0)
        mal_grads_mean_norm = torch.linalg.norm(mal_grads_mean)

        f = len(mal_users)
        rand_vecs = torch.randn_like(mal_users_grads)      # [f, d]
        zero_sum_vecs = rand_vecs - rand_vecs.mean(dim=0, keepdim=True)

        target_norm = 0.1 * mal_grads_mean_norm
        norms = torch.linalg.norm(zero_sum_vecs, dim=1, keepdim=True)
        safe = norms != 0
        scaled_vecs = torch.where(safe, zero_sum_vecs * (target_norm / norms.clamp(min=1e-12)), zero_sum_vecs)

        for usr, new_grad in zip(mal_users, scaled_vecs):
            usr.submit = new_grad.clone()


class MimicVariant(object):
    needs_own_grads = False 

    def __init__(self, warmup, argmax=True):
        self.warmup = warmup
        self.argmax = argmax

        self.t = 0
        self.mu = None
        self.z = None
        self.target_index = None

    def _get_honest_grads(self, all_users):
        grads = []
        indices = []
        for i, usr in enumerate(all_users):
            if not usr.is_malicious:
                grads.append(usr.grads)
                indices.append(i)
        return indices, grads

    def _init_z(self, grads, avg):
        self.mu = avg.clone()

        g = torch.Generator(device=avg.device)
        g.manual_seed(0)
        self.z = torch.randn(len(avg), generator=g, device=avg.device)

        cumu = torch.zeros_like(avg)
        for grad in grads:
            diff = grad - self.mu
            w = torch.dot(diff, self.z)
            cumu += w * diff

        norm = torch.linalg.norm(cumu)
        if norm != 0:
            self.z = cumu / norm

    def _update_z(self, grads, avg):
        self.mu = (self.t / (1 + self.t)) * self.mu + avg / (1 + self.t)

        cumu = torch.zeros_like(avg)
        for grad in grads:
            diff = grad - self.mu
            w = torch.dot(diff, self.z)
            cumu += w * diff

        norm = torch.linalg.norm(cumu)
        if norm != 0:
            cumu = cumu / norm

        self.z = (self.t / (1 + self.t)) * self.z + cumu / (1 + self.t)

        norm_z = torch.linalg.norm(self.z)
        if norm_z != 0:
            self.z = self.z / norm_z

    def _select_target(self, grads):
        best_val = None
        best_grad = None
        best_idx = None

        for i, grad in enumerate(grads):
            val = torch.dot(grad, self.z)
            if self.argmax:
                if best_val is None or val > best_val:
                    best_val, best_grad, best_idx = val, grad, i
            else:
                if best_val is None or val < best_val:
                    best_val, best_grad, best_idx = val, grad, i
        return best_val, best_idx, best_grad

    def attack(self, mal_users, all_users):
        if len(mal_users) == 0:
            return

        honest_indices, honest_grads = self._get_honest_grads(all_users)
        if len(honest_grads) == 0:
            return

        avg = torch.stack(honest_grads).mean(dim=0)

        if self.t == 0:
            self._init_z(honest_grads, avg)
        elif self.t < self.warmup:
            self._update_z(honest_grads, avg)

        if self.t < self.warmup:
            _, idx, selected_grad = self._select_target(honest_grads)
            self.target_index = honest_indices[idx]
            attack_grad = selected_grad
        else:
            attack_grad = all_users[self.target_index].grads

        for usr in mal_users:
            usr.submit = attack_grad.clone()

        self.t += 1


class _MinDistAttack(object):

    needs_own_grads = False  
    def __init__(self, predfns, dfns, fhat, dev_types=('unit_vec', 'sign', 'std')):
        self.predfns = predfns
        self.dfns = dfns
        self.fhat = fhat
        self.dev_types = dev_types
        self.grads_mean = None

    def _bound(self, sq_distances):
        raise NotImplementedError

    def _cost(self, sq_distances):
        raise NotImplementedError

    def cal_l2_dist(self, clients_grads, f, center, honest_mask):
        current_grads = predefenses.predefend[self.predfns](clients_grads, len(clients_grads), f)
        current_grads = defenses.defend[self.dfns](current_grads, len(current_grads), f, center, honest_mask)
        return torch.linalg.norm(current_grads - self.grads_mean).item()

    def _deviation(self, dev_type, all_updates, model_re):
        if dev_type == 'unit_vec':
            norm = torch.linalg.norm(model_re)
            return model_re / norm if norm != 0 else model_re
        if dev_type == 'sign':
            return torch.sign(model_re)
        if dev_type == 'std':
            return all_updates.std(dim=0)
        raise ValueError("Unknown dev_type")

    def _malicious_update(self, all_updates, model_re, deviation, bound):
        lamda = 10.0
        lamda_fail = lamda
        lamda_succ = 0.0
        threshold_diff = 1e-5

        while abs(lamda_succ - lamda) > threshold_diff:
            mal_update = model_re - lamda * deviation
            cost = self._cost(torch.linalg.norm(all_updates - mal_update, dim=1) ** 2).item()
            if cost <= bound:
                lamda_succ = lamda
                lamda = lamda + lamda_fail / 2
            else:
                lamda = lamda - lamda_fail / 2
            lamda_fail = lamda_fail / 2

        return model_re - lamda_succ * deviation

    def attack(self, mal_users, all_users):
        if len(mal_users) == 0:
            return

        hon_grads = [usr.grads for usr in all_users if not usr.is_malicious]
        if len(hon_grads) == 0:
            return

        all_updates = torch.stack(hon_grads, dim=0)
        model_re = all_updates.mean(dim=0)
        self.grads_mean = model_re

        bound = self._bound(torch.cdist(all_updates, all_updates, compute_mode='donot_use_mm_for_euclid_dist') ** 2).item()

        honest_mask = _honest_first_mask(len(all_updates), len(mal_users), all_updates.device)
        best_dev, max_dist = None, -1.0
        for dev_type in self.dev_types:
            candidate = self._malicious_update(all_updates, model_re, self._deviation(dev_type, all_updates, model_re), bound)
            all_grads = torch.vstack((all_updates, candidate.repeat(len(mal_users), 1)))
            dist = self.cal_l2_dist(all_grads, self.fhat, all_users[0].defense_center, honest_mask)
            if best_dev is None or dist > max_dist:
                max_dist = dist
                best_dev, mal_update = dev_type, candidate
        print("chosen perturbation is:", best_dev)

        for usr in mal_users:
            usr.submit = mal_update.clone()

class MinMax(_MinDistAttack):
    def _bound(self, sq_distances):
        return sq_distances.max()

    def _cost(self, sq_distances):
        return sq_distances.max()


class MinSum(_MinDistAttack):

    def _bound(self, sq_distances):
        return sq_distances.sum(dim=1).max()

    def _cost(self, sq_distances):
        return sq_distances.sum()
