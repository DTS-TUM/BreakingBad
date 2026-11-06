import torch
import torch.nn as nn

import datasets
import defenses as defenses
import predefenses as predefenses
import client
import models

class Server:
    def __init__(self, clients, est_malicious_proportion, batch_size, learning_rate, data_source, device,
                 img_size=112, attacker=None):
        self.clients = clients
        self.data_source = data_source
        self.est_mal_prop = est_malicious_proportion
        self.learning_rate = learning_rate
        self.device = device
        is_deit = models.uses_deit(data_source)  
        self.use_amp = is_deit and torch.device(device).type == 'cuda'
        self.data_set = datasets.build_dataset(data_source, train=False)

        self.test_net = models.build_model(data_source, pretrained=True, img_size=img_size).to(device)

        eval_batch_size = max(batch_size, 512) if is_deit else batch_size
        self.test_loader = torch.utils.data.DataLoader(self.data_set, batch_size=eval_batch_size, shuffle=False,
                                                       pin_memory=(torch.device(device).type == 'cuda'))
        self.current_weights = client.flatten_params(self.test_net).clone() 
        self.clients_grads = torch.empty((len(clients), self.current_weights.numel()),
                                         device=device, dtype=self.current_weights.dtype)
        self.defense_center = torch.zeros_like(self.current_weights)
        self.honest_mask = torch.tensor([not usr.is_malicious for usr in clients], dtype=torch.bool, device=device)
        self.skip_malicious_training = (attacker is not None
                                        and not getattr(attacker, 'needs_own_grads', True)
                                        and bool(self.honest_mask.any()))

    def dispatch_weights(self, cur_round):
        for usr in self.clients:
            if usr.is_malicious and self.skip_malicious_training:
                usr.submit = None
            else:
                usr.step(self.current_weights, self.learning_rate)
            usr.defense_center = self.defense_center 

    def collect_gradients(self):
        for idx, usr in enumerate(self.clients):
            if usr.submit is None:
                raise RuntimeError(f"client {usr.client_id} did not train and the attack gave it no update")
            self.clients_grads[idx, :] = usr.submit

    def defend(self, predefence_method, defence_method, cur_round):
        current_grads = predefenses.predefend[predefence_method](self.clients_grads, len(self.clients), int(round(len(self.clients)*self.est_mal_prop)))
        current_grads = defenses.defend[defence_method](current_grads, len(current_grads), int(round(len(current_grads)*self.est_mal_prop)), self.defense_center, self.honest_mask)
        self.defense_center = current_grads.detach().clone()
        self.current_weights -= self.learning_rate*current_grads

    def test(self):
        client.row_into_parameters(self.current_weights, self.test_net)
        total_loss = 0.0
        total_chars = 0
        correct = 0
        self.test_net.eval()
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=self.use_amp):
                    net_out = self.test_net(data)
                net_out = net_out.float()
                loss = nn.functional.cross_entropy(net_out, target, reduction='sum')
                pred = net_out.argmax(dim=1)
                correct += (pred == target).sum().item()
                total_chars += target.size(0)

                total_loss += loss.item()

        if self.data_source == 'Shakespeare':
            avg_loss = total_loss / total_chars
            perplexity = torch.exp(torch.tensor(avg_loss))
            return avg_loss, correct, perplexity.item(), total_chars
        else:
            avg_loss = total_loss / total_chars
            return avg_loss, correct, total_chars