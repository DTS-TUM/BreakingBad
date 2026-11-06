import torch
import torch.nn as nn
import torch.optim as optim
import datasets
import models

DEIT_WEIGHT_DECAY = 0.0   
_SHARED_NETS = {}

def _shared_net(data_source, img_size, device):
    key = (data_source, img_size, str(device))
    if key not in _SHARED_NETS:
        rng_state = torch.get_rng_state()
        net = models.build_model(data_source, pretrained=False, img_size=img_size)
        torch.set_rng_state(rng_state)
        stateful = sorted({type(m).__name__ for m in net.modules() if isinstance(m, nn.modules.batchnorm._NormBase)})
        if stateful:
            raise ValueError(f"{data_source} model has layers with running statistics {stateful}; "
                             "it cannot be shared between clients")
        _SHARED_NETS[key] = net.to(device)
    return _SHARED_NETS[key]

def flatten_params(net):
    return torch.cat([p.detach().reshape(-1) for p in net.parameters()])

def flatten_grads(net):
    return torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).detach().reshape(-1)
                      for p in net.parameters()])

def row_into_parameters(row, net):
    offset = 0
    for param in net.parameters():
        new_size = param.numel()
        param.data.copy_(row[offset:offset + new_size].view_as(param))
        offset += new_size

class Client:
    def __init__(self, client_id, local_iter, batch_size, is_malicious, momentum, data_source, idx, device,
                 img_size=112):
        self.batch_size = batch_size
        self.is_malicious = is_malicious
        self.data_source = data_source
        self.is_deit = models.uses_deit(data_source)   
        self.client_id = client_id
        self.data_idx = idx
        self.local_iter = local_iter
        self.learning_rate = None
        self.grads = None
        self.defense_center = None 
        self.device = device
        self.use_amp = self.is_deit and torch.device(device).type == 'cuda'
        self.data_set = datasets.build_dataset(data_source, idx, train=True)

        self.net = _shared_net(data_source, img_size, device)

        self.momentum = momentum
        self.batches = datasets.DeviceBatches(self.data_set, data_source, batch_size, device)
        if self.is_deit:
            self.optimizer = optim.SGD(self.net.parameters(), lr=0.0, weight_decay=DEIT_WEIGHT_DECAY)
        elif data_source == 'CIFAR10':
            self.optimizer = optim.SGD(self.net.parameters(), lr=0.0, weight_decay=1e-2)
        else:
            self.optimizer = optim.SGD(self.net.parameters(), lr=0.0)
        self.submit = None
        self._grad_sum = None   

    def train(self, data, target, apply_step=True):
        self.optimizer.zero_grad()
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=self.use_amp):
            net_out = self.net(data)
        loss = nn.functional.cross_entropy(net_out.float(), target)

        loss.backward()
        if self.is_deit:
            g = flatten_grads(self.net)
            if self._grad_sum is None:
                self._grad_sum = g
            else:
                self._grad_sum.add_(g)
        if apply_step:
            self.optimizer.step()

    def step(self, current_params, learning_rate):
        row_into_parameters(current_params, self.net)
        self.net.train()  
        for group in self.optimizer.param_groups:
            group['lr'] = learning_rate

        self._grad_sum = None
        for iteration in range(self.local_iter):
            data, target = self.batches.next()
            self.train(data, target, apply_step=not (self.is_deit and iteration == self.local_iter - 1))

        if self.is_deit:
            grads = self._grad_sum
            self._grad_sum = None
        else:
            grads = (current_params - flatten_params(self.net)) / learning_rate

        if self.grads is None:
            self.grads = (1 - self.momentum) * grads
        else:
            self.grads = self.momentum * self.grads + (1 - self.momentum) * grads

        self.submit = self.grads   
        self.optimizer.zero_grad(set_to_none=True)