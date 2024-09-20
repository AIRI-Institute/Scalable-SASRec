from time import time
from functools import reduce
import uuid
from tqdm import tqdm
import os

from clearml import Task, Logger
from omegaconf import OmegaConf
import hydra

import numpy as np
import torch
from torch.utils.data import DataLoader

from model import SASRec
from data import get_dataset, data_to_sequences, SequentialDataset
from utils import topn_recommendations, downvote_seen_items
from eval_utils import model_evaluate, sasrec_model_scoring, get_test_scores


@hydra.main(version_base=None, config_path="configs", config_name="train")
def main(config):
    
    print(OmegaConf.to_yaml(config))
    
    if hasattr(config, 'cuda_visible_devices'):
        os.environ['CUDA_VISIBLE_DEVICES'] = str(config.cuda_visible_devices)
    
    if hasattr(config, 'project_name'):
        Task.set_random_seed(config.trainer_params.seed)
        task = Task.init(project_name=config.project_name, task_name=config.task_name,
                        reuse_last_task_id=False)
        task.connect(OmegaConf.to_container(config))
    else:
        task = None
        
        
    base_config = dict(
        num_epochs = config.trainer_params.num_epochs,
        maxlen = config.model_params.maxlen,
        hidden_units = config.model_params.hidden_units,
        dropout_rate = config.model_params.dropout_rate,
        num_blocks = config.model_params.num_blocks,
        num_heads = config.model_params.num_heads,
        batch_size = config.dataloader.batch_size,
        learning_rate = config.trainer_params.learning_rate,
        fwd_type = config.model_params.fwd_type,
        l2_emb = 0,
        n_neg_samples = config.dataloader.n_neg_samples,
        manual_seed = config.trainer_params.seed,
        sampler_seed = config.trainer_params.seed,
        sampling = config.model_params.sampling,
        patience = config.trainer_params.patience,
        skip_epochs = config.trainer_params.skip_epochs
    )
    
    if config.model_params.fwd_type == 'gbce':
        base_config['gbce_t'] = config.model_params.gbce_t
        
    if config.model_params.fwd_type == 'sce':
        base_config['n_buckets'] = config.model_params.n_buckets
        base_config['bucket_size_x'] = config.model_params.bucket_size_x
        base_config['bucket_size_y'] = config.model_params.bucket_size_y
        base_config['mix_x'] = config.model_params.mix_x

    device = 'cuda'
    training, data_description, _, testset_valid, testset_, holdout_valid, holdout_ = get_dataset(path=config.data_path, splitting=config.splitting)

    if task:
        log = Logger.current_logger()
    else:
        log = None

    model = \
        build_sasrec_model(base_config, training, data_description,
                           testset_valid=testset_valid, holdout_valid=holdout_valid, device=device,
                           task=task, log=log)

    test_scores = get_test_scores(model, data_description, testset_, holdout_, device)
    test_scores_meta = reduce(lambda s, metric_name: s + f'\n{metric_name}:{test_scores[metric_name]:.3g}', test_scores.keys(), '')

    print(test_scores_meta)
            
    if task:
        for metric_name, metric_value in test_scores.items():
            log.report_single_value(name=f'test_{metric_name}', value=round(metric_value, 4))
        
        task.close()
        

def set_worker_random_state(id):
    dataset = torch.utils.data.get_worker_info().dataset
    dataset.seed = dataset.seed + id
    dataset.random_state = np.random.RandomState(dataset.seed)


def prepare_sasrec_model(config, data, data_description, device):
    n_users = data_description['n_users']
    n_items = data_description['n_items']
    model = SASRec(n_items, config).to(device)

    train_sequences = data_to_sequences(data, data_description)

    sampler = \
        DataLoader(SequentialDataset(train_sequences, n_users, n_items,
        maxlen = config['maxlen'],
        seed = config['sampler_seed'],
        n_neg_samples = config['n_neg_samples'],
        pad_token = model.pad_token,
        sampling = config['sampling']), batch_size=config['batch_size'], shuffle=True, num_workers=8, pin_memory=True, prefetch_factor=10, worker_init_fn=set_worker_random_state, persistent_workers=True, drop_last=True)
    
    n_batches = len(train_sequences) // config['batch_size']
    
    optimizer = \
    torch.optim.Adam(model.parameters(),
        lr = config['learning_rate'],
        betas = (0.9, 0.98))
    
    return model, sampler, n_batches, optimizer


def train_sasrec_epoch(model, num_batch, l2_emb, sampler, optimizer, device):
    model.train()
    pad_token = model.pad_token
    losses = []
    for _, *seq_data in sampler:
        # convert batch data into torch tensors
        seq, pos, neg = (torch.tensor(np.array(x), device=device, dtype=torch.long) for x in seq_data)
        loss = model(seq, pos, neg)
        optimizer.zero_grad()
        if l2_emb != 0:
            for param in model.item_emb.parameters():
                loss += l2_emb * torch.norm(param)**2
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return losses


def build_sasrec_model(config, data, data_description, testset_valid, holdout_valid, device, task, log):
    '''Simple MF training routine without early stopping'''
    model, sampler, n_batches, optimizers = prepare_sasrec_model(config, data, data_description, device)
    losses = {}
    metrics = {}
    ndcg = {}
    best_ndcg = 0
    wait = 0

    start_time = time()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start_memory = torch.cuda.memory_allocated()

    checkpt_name = uuid.uuid4().hex
    if not os.path.exists('./checkpt'):
        os.mkdir('./checkpt')
    
    checkpt_path = os.path.join('./checkpt', f'{checkpt_name}.chkpt')

    for epoch in tqdm(range(config['num_epochs'])):
        losses[epoch] = train_sasrec_epoch(
            model, n_batches, config['l2_emb'], sampler, optimizers, device
        )
        if epoch % config['skip_epochs'] == 0:
            val_scores = sasrec_model_scoring(model, testset_valid, data_description, device)
            downvote_seen_items(val_scores, testset_valid, data_description)
            val_recs = topn_recommendations(val_scores, topn=10)
            val_metrics = model_evaluate(val_recs, holdout_valid, data_description)
            metrics[epoch] = val_metrics
            ndcg_ = val_metrics['ndcg@10']
            ndcg[epoch] = ndcg_
            
            if task and (epoch % 5 == 0):
                log.report_scalar("Loss", series='Val', iteration=epoch, value=np.mean(losses[epoch]))
                log.report_scalar("NDCG", series='Val', iteration=epoch, value=ndcg_)

            if ndcg_ > best_ndcg:
                best_ndcg = ndcg_
                torch.save(model.state_dict(), checkpt_path)
                wait = 0
            elif wait < config['patience'] // config['skip_epochs'] + 1:
                wait += 1
            else:
                break
    
    torch.cuda.synchronize()
    training_time_sec = time() - start_time
    full_peak_training_memory_bytes = torch.cuda.max_memory_allocated()
    peak_training_memory_bytes = torch.cuda.max_memory_allocated() - start_memory
    training_epoches = len(losses)
    
    model.load_state_dict(torch.load(checkpt_path))
    os.remove(checkpt_path)

    print('Peak training memory, mb:', round(full_peak_training_memory_bytes/ 1024. / 1024., 2))
    print('Training epoches:', training_epoches)
    print('Training time, m:', round(training_time_sec/ 60., 2))
    
    if task:
        ind_max = np.argmax(list(ndcg.values())) * config['skip_epochs']
        for metric_name, metric_value in metrics[ind_max].items():
            log.report_single_value(name=f'val_{metric_name}', value=round(metric_value, 4))
        log.report_single_value(name='train_peak_mem_mb', value=round(peak_training_memory_bytes/ 1024. / 1024., 2))
        log.report_single_value(name='full_train_peak_mem_mb', value=round(full_peak_training_memory_bytes/ 1024. / 1024., 2))
        log.report_single_value(name='train_epoches', value=training_epoches)
        log.report_single_value(name='train_time_m', value=round(training_time_sec/ 60., 2))

    return model


if __name__ == "__main__":

    main()
