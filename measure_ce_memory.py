import argparse
import pickle as pkl
import os

import torch
import torch.nn.functional as F

from model import SASRecBackBone


class Hooker():
  def __init__(self, mem):
    self.mem = mem

  def __call__(self, x):
    torch.cuda.synchronize()
    self.mem['post_ce_bwd'] = (torch.cuda.memory_allocated(), torch.cuda.max_memory_allocated())
    torch.cuda.reset_peak_memory_stats()


class CESASRec(SASRecBackBone):
    def __init__(self, item_num, config):
        super().__init__(item_num, config)
        self.mem_stats = {'input': None, 'post_model_init': None, 'post_grad_init': None, 'post_opt_init': None,
                          'pre_f': None, 'pre_ce': None, 'post_ce': None,
                          'post_ce_bwd': None,  'post_bwd': None}

    def forward(self, log_seqs, pos_seqs):
        torch.cuda.synchronize()
        self.mem_stats['pre_f'] = (torch.cuda.memory_allocated(), torch.cuda.max_memory_allocated())
        torch.cuda.reset_peak_memory_stats()

        emb = self.log2feats(log_seqs)
        emb.register_hook(Hooker(self.mem_stats))

        torch.cuda.synchronize()
        self.mem_stats['pre_ce'] = (torch.cuda.memory_allocated(), torch.cuda.max_memory_allocated())
        torch.cuda.reset_peak_memory_stats()

        logits = emb @ self.item_emb.weight.T
        indices = torch.where(pos_seqs.view(-1) != self.pad_token)
        loss = F.cross_entropy(logits.view(-1, logits.shape[-1])[indices], pos_seqs.view(-1)[indices], reduction='mean')

        torch.cuda.synchronize()
        self.mem_stats['post_ce'] = (torch.cuda.memory_allocated(), torch.cuda.max_memory_allocated())
        torch.cuda.reset_peak_memory_stats()

        return loss


def measure_model(catalog_size, batch_size):
    base_config = dict(
        num_epochs = -1,
        maxlen = 200,
        hidden_units = 64,
        dropout_rate = 0.3,
        num_blocks = 2,
        num_heads = 1,
        batch_size = batch_size,
        learning_rate = 0.001,
        l2_emb = 0,
        n_neg_samples = 1,
        manual_seed = 37,
        use_sparse_emb = False
    )

    seq = torch.randint(10, catalog_size-10, (batch_size, 200), device='cuda')
    pos = torch.randint(10, catalog_size-10, (batch_size, 200), device='cuda')

    torch.cuda.synchronize()
    input_mem_stats = (torch.cuda.memory_allocated(), torch.cuda.max_memory_allocated())
    torch.cuda.reset_peak_memory_stats()

    model = CESASRec(catalog_size, base_config).cuda()

    model.mem_stats['input'] = input_mem_stats

    torch.cuda.synchronize()
    model.mem_stats['post_model_init'] = (torch.cuda.memory_allocated(), torch.cuda.max_memory_allocated())
    torch.cuda.reset_peak_memory_stats()

    for param in model.parameters():
        param.grad = torch.zeros_like(param)


    torch.cuda.synchronize()
    model.mem_stats['post_grad_init'] = (torch.cuda.memory_allocated(), torch.cuda.max_memory_allocated())
    torch.cuda.reset_peak_memory_stats()

    optimizer = torch.optim.Adam(model.parameters())
    optimizer.step()

    torch.cuda.synchronize()
    model.mem_stats['post_opt_init'] = (torch.cuda.memory_allocated(), torch.cuda.max_memory_allocated())
    torch.cuda.reset_peak_memory_stats()

    try:
        loss = model(seq, pos)
        loss.backward()
        torch.cuda.synchronize()
        model.mem_stats['post_bwd'] = (torch.cuda.memory_allocated(), torch.cuda.max_memory_allocated())
        torch.cuda.reset_peak_memory_stats()
        
    except torch.cuda.OutOfMemoryError:
        pass

    return model.mem_stats


if __name__ == '__main__':
    parser = argparse.ArgumentParser('')
    parser.add_argument('--bs', type=int, required=True)
    parser.add_argument('--catalog', type=int, required=True)
    args = parser.parse_args()

    measurement = measure_model(catalog_size=args.catalog, batch_size=args.bs)
    print(measurement)
    
    if not os.path.exists('mem_data'):
        os.mkdir('mem_data')

    with open(f'mem_data/{args.catalog}_cat_{args.bs}_bs', 'wb') as f:
        pkl.dump(measurement, f)
