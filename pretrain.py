import shutil
import time
import numpy as np
import socket
from modelwrapper import ModelWrapper
from transformers import AutoTokenizer
from datasets import load_from_disk
import os
import torch
from evaluate_icl import icl_main
import pickle as pkl

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def main(model_name, n_gpus, train_data_fname, val_data_fname, num_train_steps, num_warmup_steps, per_device_batch_size, effective_bsz, log_every_steps, lr, bf16, patience_num_train_steps, patience_perplexity_decrease, out_dir):
    save_model_ckpt_steps = list(range(0, num_train_steps+1, log_every_steps))
    lr_scheduler = 'linear'
    flash_attention = True
    gradient_checkpointing = True
    train_data = load_from_disk(train_data_fname)
    val_data = load_from_disk(val_data_fname)
    mw = ModelWrapper(model_type='clm', model_name=model_name, tokenizer=AutoTokenizer.from_pretrained(model_name),
                      load_pretrained=True, load_pretrained_hf_name=model_name, bf16=bf16)
    mw.train(is_lm=True, train_data=train_data, val_data=val_data, shuffle_train=False,
             n_gpus=n_gpus, num_train_steps=num_train_steps, min_num_train_steps=500, lr=lr, lr_scheduler=lr_scheduler,
             num_warmup_steps=num_warmup_steps, nonbias_weight_decay=0,
             per_device_batch_size=per_device_batch_size,
             effective_bsz=effective_bsz,
             use_flash_attention=flash_attention, gradient_checkpointing=gradient_checkpointing,
             save_model_ckpts=True, save_model_ckpts_steps=save_model_ckpt_steps,
             early_stopping=True, patience_num_train_steps=patience_num_train_steps, patience_perplexity_decrease=patience_perplexity_decrease,
             output_dir=out_dir)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    # data
    parser.add_argument("--train_data_fname", type=str, default=None)
    parser.add_argument("--val_data_fname", type=str, default=None)
    # training
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--n_gpus", type=int, default=None)
    parser.add_argument("--num_train_steps", type=int, default=None)
    parser.add_argument("--num_warmup_steps", type=int, default=None)
    parser.add_argument("--per_device_batch_size", type=int, default=None)
    parser.add_argument("--effective_bsz", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--bf16", type=int, default=0)
    parser.add_argument("--patience_num_train_steps", type=int)
    parser.add_argument("--patience_perplexity_decrease", type=float, default=None)
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--log_every_steps", type=int, default=None)

    args = parser.parse_args()
    
    save_model_ckpt_steps = list(range(0, args.num_train_steps+1, args.log_every_steps))
    lr_scheduler = 'linear'
    flash_attention = True
    gradient_checkpointing = True
    train_data = load_from_disk(args.train_data_fname)
    val_data = load_from_disk(args.val_data_fname)
    mw = ModelWrapper(model_type='clm', model_name=args.model_name, tokenizer=AutoTokenizer.from_pretrained(args.model_name),
                      load_pretrained=True, load_pretrained_hf_name=args.model_name, bf16=args.bf16)
    mw.train(is_lm=True, train_data=train_data, val_data=val_data, shuffle_train=False,
             n_gpus=args.n_gpus, num_train_steps=args.num_train_steps, min_num_train_steps=500, lr=args.lr, lr_scheduler=lr_scheduler,
             num_warmup_steps=args.num_warmup_steps, nonbias_weight_decay=0,
             per_device_batch_size=args.per_device_batch_size,
             effective_bsz=args.effective_bsz,
             use_flash_attention=flash_attention, gradient_checkpointing=gradient_checkpointing,
             save_model_ckpts=True, save_model_ckpts_steps=save_model_ckpt_steps,
             early_stopping=True, patience_num_train_steps=args.patience_num_train_steps, patience_perplexity_decrease=args.patience_perplexity_decrease,
             output_dir=args.out_dir)