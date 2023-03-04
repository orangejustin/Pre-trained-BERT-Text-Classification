import os, sys, pdb
import numpy as np
import random
import torch

import math

from tqdm import tqdm as progress_bar
import umap.umap_ as umap

from utils import set_seed, setup_gpus, check_directories
from dataloader import get_dataloader, check_cache, prepare_features, process_data, prepare_inputs
from load import load_data, load_tokenizer
from arguments import params
from model import IntentModel, SupConModel, CustomModel
from torch import nn
from transformers import AdamW, get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup
import matplotlib.pyplot as plt

device='cuda'

def baseline_train(args, model, datasets, tokenizer):
    criterion = nn.CrossEntropyLoss()  # combines LogSoftmax() and NLLLoss()
    # task1: setup train dataloader
    train_dataloader = get_dataloader(args, datasets["train"], "train")
    
    # task2: setup model's optimizer_scheduler if you have
    
    # task3: write a training loop
    for epoch_count in range(args.n_epochs):
        losses = 0
        model.train()
        
        for step, batch in progress_bar(enumerate(train_dataloader), total=len(train_dataloader)):
            inputs, labels = prepare_inputs(batch, model, use_text=False)
            logits = model(inputs, labels)
            loss = criterion(logits, labels)
            loss.backward()

            model.optimizer.step()  # backprop to update the weights
#             model.scheduler.step()  # Update learning rate schedule
            model.optimizer.zero_grad()
            losses += loss.item()
        run_eval(args, model, datasets, tokenizer, split='validation')
    run_eval(args, model, datasets, tokenizer, split='test')
    print('epoch', epoch_count, '| losses:', losses / len(train_dataloader))

def custom_train(args, model, datasets, tokenizer):
    criterion = nn.CrossEntropyLoss()  # combines LogSoftmax() and NLLLoss()
    # task1: setup train dataloader
    train_dataloader = get_dataloader(args, datasets["train"], "train")
    n_epoch = args.n_epochs
    n_train = n_epoch * len(train_dataloader)
    n_warm_up = args.warm_up_step
    # task2: setup model's optimizer_scheduler if you have
    if args.scheduler == "linear":
        model.scheduler = get_linear_schedule_with_warmup(
            model.optimizer,
            num_warmup_steps=n_warm_up,
            num_training_steps=n_train
        )

    if args.scheduler == "cosine":
        model.scheduler = get_cosine_schedule_with_warmup(
            model.optimizer,
            num_warmup_steps=n_warm_up,
            num_training_steps=n_train
        )
    
    # task3: write a training loop
    for epoch_count in range(args.n_epochs):
        losses = 0
        model.train()

        for step, batch in progress_bar(enumerate(train_dataloader), total=len(train_dataloader)):
            inputs, labels = prepare_inputs(batch, model, use_text=False)
            logits = model(inputs, labels)
            loss = criterion(logits, labels)
            loss.backward()
            model.optimizer.step()
            model.optimizer.zero_grad()

            if model.scheduler: model.scheduler.step()
            losses += loss.item()
        run_eval(args, model, datasets, tokenizer, split='validation')
    run_eval(args, model, datasets, tokenizer, split='test')
    print('epoch', epoch_count, '| losses:', losses / len(train_dataloader))

def run_eval(args, model, datasets, tokenizer, split='validation'):
    model.eval()
    dataloader = get_dataloader(args, datasets[split], split)

    acc = 0
    for step, batch in progress_bar(enumerate(dataloader), total=len(dataloader)):
        inputs, labels = prepare_inputs(batch, model)
        logits = model(inputs, labels)
        
        tem = (logits.argmax(1) == labels).float().sum()
        acc += tem.item()
    
    print(f'{split} acc:', acc/len(datasets[split]), f'|dataset split {split} size:', len(datasets[split]))
    return f'{split} acc:', acc/len(datasets[split]), f'|dataset split {split} size:', len(datasets[split])

def supcon_train(args, model, datasets, tokenizer):
    from loss import SupConLoss
    criterion = SupConLoss(temperature=args.temperature)
    # task1: load training split of the dataset
    train_dataloader = get_dataloader(args, datasets["train"], "train")
    n_epoch = args.n_epochs
    n_train = n_epoch * len(train_dataloader)
    n_warm_up = args.warm_up_step
    # task2: setup optimizer_scheduler in your model
    model.scheduler = get_cosine_schedule_with_warmup(
        model.optimizer,
        num_warmup_steps=n_warm_up,
        num_training_steps=n_train
    )

    # task3: write a training loop for SupConLoss function 
    for epoch_count in range(args.n_epochs):
        losses = 0
        model.train()

        for step, batch in progress_bar(enumerate(train_dataloader), total=len(train_dataloader)):
            inputs, labels = prepare_inputs(batch, model, use_text=False)
            drop_logit1 = model(inputs, labels)
            drop_logit2 = model(inputs, labels)

            logits = torch.cat([drop_logit1.unsqueeze(1), drop_logit2.unsqueeze(1)], dim=1)
            if args.task == 'supcon':
                loss = criterion(logits, labels)
            elif args.task == 'simclr':
                loss = criterion(logits)
            
            model.optimizer.zero_grad()
            loss.backward()

            model.optimizer.step()  # backprop to update the weights
            model.scheduler.step()
            losses += loss.item()

        print('epoch', epoch_count, '| losses:', losses/len(train_dataloader))

def plot(args, model, datasets, name):
    emb_lst, label_lst = [], []
    model.eval()
    dataloader = get_dataloader(args, datasets['test'], 'test')

    for step, batch in progress_bar(enumerate(dataloader), total=len(dataloader)):
        inputs, labels = prepare_inputs(batch, model, use_text=False)
        logit = model(inputs, labels)
        emb, label = logit[labels < 10], labels[labels < 10]
        
        emb_lst += list(map(lambda x: x.tolist(), emb))
        label_lst += list(map(lambda x: x.item(), label))
        
    print("emb_lst size", len(emb_lst))
    print("label_lst size", len(label_lst))
    reducer = umap.UMAP()
    embedding = reducer.fit_transform(emb_lst)

    plt.scatter(embedding[:, 0], embedding[:, 1], c=label_lst, cmap='Spectral', s=10)
#     plt.gca().set_aspect('equal', 'datalim')
#     plt.colorbar(boundaries=np.arange(11)-0.5).set_ticks(np.arange(10))
    plt.title('UMAP projection of the %s Embedding'%name, fontsize=24)
    plt.show()
    # plt.savefig("%s_umap.png"%name)
    
    
if __name__ == "__main__":
    args = params()
    args = setup_gpus(args)
    args = check_directories(args)
    set_seed(args)

    cache_results, already_exist = check_cache(args)
    tokenizer = load_tokenizer(args)

    if already_exist:
        features = cache_results
    else:
        data = load_data()
        features = prepare_features(args, data, tokenizer, cache_results)
    datasets = process_data(args, features, tokenizer)
    for k,v in datasets.items():
        print(k, len(v))
 
    if args.task == 'baseline':
        model = IntentModel(args, tokenizer, target_size=60).to(device)
        run_eval(args, model, datasets, tokenizer, split='validation')
        run_eval(args, model, datasets, tokenizer, split='test')
        baseline_train(args, model, datasets, tokenizer)
        run_eval(args, model, datasets, tokenizer, split='test')
        plot(args, model, datasets, "Baseline")
    elif args.task == 'custom': # you can have multiple custom task for different techniques
        model = CustomModel(args, tokenizer, target_size=60).to(device)
        run_eval(args, model, datasets, tokenizer, split='validation')
        run_eval(args, model, datasets, tokenizer, split='test')
        custom_train(args, model, datasets, tokenizer)
        run_eval(args, model, datasets, tokenizer, split='test')
    elif args.task == 'supcon':
        model = SupConModel(args, tokenizer, target_size=60).to(device)
        supcon_train(args, model, datasets, tokenizer)
        run_eval(args, model, datasets, tokenizer, split='test')
        plot(args, model, datasets, "SupCon")
    elif args.task == 'simclr':
        model = SupConModel(args, tokenizer, target_size=60).to(device)
        supcon_train(args, model, datasets, tokenizer)
        run_eval(args, model, datasets, tokenizer, split='test')
        # plot(args, model, datasets, "SupCon")
        plot(args, model, datasets, "SimCLR")
