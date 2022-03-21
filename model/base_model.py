import os
import torch


class BaseModel():
    def name(self):
        return 'BaseModel'

    def initialize(self, opt):
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        self.Tensor = torch.cuda.FloatTensor if self.gpu_ids else torch.Tensor
        self.save_dir = os.path.join(opt.checkpoints_dir)
#         self.device = torch.device('cuda:%d'%self.gpu_ids[0] if self.gpu_ids is not None else 'cpu')
        self.device = torch.device('cuda' if self.gpu_ids is not None else 'cpu')

    def set_input(self, input):
        self.input = input

    def forward(self):
        pass

    # used in test time, no backprop
    def test(self):
        pass

    def get_image_paths(self):
        pass

    def optimize_parameters(self):
        pass

    def get_current_visuals(self):
        return self.input

    def get_current_errors(self):
        return {}

    def save(self, label):
        pass

    # helper saving function that can be used by subclasses
    def save_network(self, network, network_label, epoch_label, gpu_ids, device):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        torch.save(network.cpu().module.state_dict(), save_path)
        if len(gpu_ids) and torch.cuda.is_available():
            network.to(device)

    # helper loading function that can be used by subclasses
    def load_network(self, network, network_label, iter_label):
        save_filename = '%s_state_%s.pth' % (iter_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        states = torch.load(save_path, map_location='cpu')
        net_state = {}
        for key, value in states['network'].items():
            if 'module.' in key:
                net_state[key.replace('module.', '')] = value
            else:
                net_state[key] = value
        network.load_state_dict(net_state)
        
        
    # helper saving function that can be used by subclasses
    def save_states(self, network, optimizer, scheduler, network_label, iter_label, gpu_ids, device):
        save_filename = '%s_state_%s.pth' % (iter_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        states = {}
        states['network'] = network.cpu().module.state_dict()
        states['optimizer'] = optimizer.state_dict()
        states['scheduler'] = scheduler.state_dict()
        torch.save(states, save_path)
        if len(gpu_ids):
            network.to(device)
        
    def save_states_simple(self, network, network_label, gpu_ids, device):
        save_filename = '%s_state_%s.pth' % ('best', network_label)
        save_path = os.path.join('best_ckpt', save_filename)
        states = {}
        try:
            states['network'] = network.cpu().module.state_dict()
        except:
            states['network'] = network.cpu().state_dict()
        torch.save(states, save_path)
        if len(gpu_ids):
            network.to(device)

    # helper loading function that can be used by subclasses
    def load_states(self, network, optimizer, scheduler, network_label, iter_label):
        save_filename = '%s_state_%s.pth' % (iter_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        states = torch.load(save_path, map_location='cpu')
        net_state = {}
        for key, value in states['network'].items():
            if 'module.' in key:
                net_state[key.replace('module.', '')] = value
            else:
                net_state[key] = value
        network.load_state_dict(net_state)
        optimizer.load_state_dict(states['optimizer'])
        scheduler.load_state_dict(states['scheduler'])

    def load_states_simple(self, network, network_label, iter_label):
        save_filename = '%s_state_%s.pth' % (iter_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        states = torch.load(save_path, map_location='cpu')
        net_state = {}
        for key, value in states['network'].items():
            if 'module.' in key:
                net_state[key.replace('module.', '')] = value
            else:
                net_state[key] = value
        network.load_state_dict(net_state)

    # update learning rate (called once every epoch)
    def update_learning_rate(self, iteration_step):
        for scheduler in self.schedulers:
            scheduler.step()
        if iteration_step % 1e4 == 0:
            lr = self.optimizers[0].param_groups[0]['lr']
            print('learning rate = %.7f' % lr)

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad
