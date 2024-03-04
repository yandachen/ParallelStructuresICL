import math
import torch
from modelwrapper import ModelWrapper
from transformers import AutoTokenizer
from datasets import load_from_disk, Dataset
import numpy as np
from copy import deepcopy
import pickle as pkl
from tqdm import trange, tqdm
import os
import multiprocessing
from functools import partial
from scipy.stats import pearsonr, spearmanr


def detect(mw, train_input_ids, eval_input_ids, num_epochs, lr, train_input_masked_tokenidxs, eval_input_masked_tokenidxs, eval_window_size, eval_stride):
	train_data = {'input_ids': torch.tensor([train_input_ids])}
	train_data['masked_tokenidxs'] = torch.tensor([train_input_masked_tokenidxs])
	val_data = {'input_ids': torch.tensor([eval_input_ids])}
	val_data['masked_tokenidxs'] = torch.tensor([eval_input_masked_tokenidxs])
	trained_token_losses = mw.train_one_example(train_lm=True, train_data=Dataset.from_dict(train_data), val_data=Dataset.from_dict(val_data),
												num_epochs=num_epochs, lr=lr, debug=False, patience=0, use_flash_attention=False, eval_window_size=eval_window_size, eval_stride=eval_stride)
	return np.array(trained_token_losses)


def detect_structure_batch_examples(exidxs, mw, data_dir, train_window_size, eval_window_size, num_epochs, lr, out_dir):
	assert eval_window_size % 2 == 0
	eval_stride = eval_window_size // 2
	# load data
	data = load_from_disk(data_dir)
	gpu_id = queue.get()
	os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
	mw.model.cuda()
	exidx2out = {}
	out_fname = f'{out_dir}/{exidxs[0]}-{exidxs[-1]}.pkl'
	if os.path.exists(out_fname):
		queue.put(gpu_id)
		return
	# placeholder so other processes won't process this batch
	pkl.dump({}, open(out_fname, 'wb'))
	for exidx in tqdm(exidxs):
		input_ids = data[exidx]['input_ids']
		num_windows = math.ceil(len(input_ids) / train_window_size)
		# evaluate perplexity without training
		orig_losses = mw.evaluate_perplexity_limit_context_size(eval_data=Dataset.from_dict({'input_ids': torch.tensor([input_ids])}), 
																window_size=eval_window_size, stride=eval_stride, per_device_batch_size=128, use_accelerator=False)['token_losses'][0]
		# train and evaluate
		# (i, j) is fine-tuned on window i and evaluated on token j
		trained_losses = np.full((num_windows-1, num_epochs, len(input_ids)), np.nan)
		# the index of the fine-tuned window
		for window_idx in range(0, num_windows-1):
			# both prefix and suffix will be entire inputs (can see the entire context)
			train_masked_tokenidxs = list(range(window_idx * train_window_size)) + list(range((window_idx+1) * train_window_size, len(input_ids)))
			eval_masked_tokenidxs = list(range((window_idx+1) * train_window_size))
			ex_mw = deepcopy(mw)
			ex_trained_ckpts_token_losses = detect(
				ex_mw, input_ids, input_ids, num_epochs, lr, train_masked_tokenidxs, eval_masked_tokenidxs, eval_window_size, eval_stride)
			trained_losses[window_idx][:, ((window_idx+1) * train_window_size):] = ex_trained_ckpts_token_losses[:, ((window_idx+1) * train_window_size):]
		# sanity check
		for window_idx in range(num_windows-1):
			assert np.all(np.isnan(trained_losses[window_idx][:, : (window_idx+1) * train_window_size]))
			# early stopping may happen
			optimal_num_epochs = 0
			while (optimal_num_epochs+1 < num_epochs) and (not np.isnan(trained_losses[window_idx][optimal_num_epochs+1][-1])):
				optimal_num_epochs += 1
			assert np.all(np.isnan(trained_losses[window_idx][optimal_num_epochs+1:]))
			assert not np.any(np.isnan(trained_losses[window_idx][:optimal_num_epochs+1, (window_idx+1) * train_window_size:]))
		# shape (num_windows-1, num_epochs, num_tokens)
		loss_decrease = orig_losses - trained_losses
		# for each num_epoch, do early stopping for EACH evaluation token separately, calculate the loss decrease for each token and the maximum windowidx for each token
		stop_numepochs2out = {}
		for stop_numepochs in range(1, num_epochs+1):
			earlystop_loss_decrease = loss_decrease[:, :stop_numepochs, :]
			max_loss_decrease = [np.nan for _ in range(train_window_size)] + np.nanmax(earlystop_loss_decrease[:, :, train_window_size:], axis=(0, 1)).tolist()
			max_loss_decrease_windowidx = [np.nan for _ in range(train_window_size)] + np.nanargmax(np.nanmax(earlystop_loss_decrease[:, :, train_window_size:], axis=1), axis=0).tolist()
			assert len(max_loss_decrease_windowidx) == len(max_loss_decrease) == len(input_ids)
			assert not np.any(np.isnan(max_loss_decrease[train_window_size:]))
			assert not np.any(np.isnan(max_loss_decrease_windowidx[train_window_size:]))
			stop_numepochs2out[stop_numepochs] = {'max_loss_decrease': max_loss_decrease,
												  'max_loss_decrease_windowidx': max_loss_decrease_windowidx,
												  'orig_losses': orig_losses, 'trained_losses': trained_losses}
		exidx2out[exidx] = {'loss_decrease': loss_decrease, 'numepochs2out': stop_numepochs2out}
	pkl.dump(exidx2out, open(out_fname, 'wb'))
	queue.put(gpu_id)


#### check for approximation
def calculate_overlap(exidx_tokenidxs1, exidx_tokenidxs2):
    exidx_tokenidxs1 = set(exidx_tokenidxs1)
    exidx_tokenidxs2 = set(exidx_tokenidxs2)
    return len(exidx_tokenidxs1.intersection(exidx_tokenidxs2)) / len(exidx_tokenidxs1)

def compare_different_params():
    expdir = f'{os.environ["ICL2"]}/parallel_structure/gpt2-seqlen1024-completewindows-2.6M/'
    gt_data = pkl.load(open(f'{expdir}/trainw1_evalw12_nepochs1_lr1e-4/dev/exidx2out.pkl', 'rb'))
    approx_data = pkl.load(open(f'{expdir}/trainw128_evalw12_nepochs1_lr1e-4/dev/exidx2out.pkl', 'rb'))
    print('data loaded...')
    gt_exidx2ld = {exidx: gt_data[exidx]['numepochs2out'][1]['max_loss_decrease'] for exidx in gt_data}
    approx_exidx2ld = {exidx: approx_data[exidx]['numepochs2out'][1]['max_loss_decrease'] for exidx in approx_data}
    name2exidx2ld = {'gt': gt_exidx2ld, 'approx': approx_exidx2ld}
    name2scores = {name: [name2exidx2ld[name][exidx][tok_pos]] for exidx in range(10000) for tok_pos in range(128, 1024) for name in name2exidx2ld}
    print(pearsonr(name2scores['gt'], name2scores['approx']))
    print(spearmanr(name2scores['gt'], name2scores['approx']))
####


if __name__ == '__main__':
	import torch.multiprocessing
	torch.multiprocessing.set_sharing_strategy('file_system')

	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument("--gpu", type=str, default=None)
	parser.add_argument("--model_name", type=str, default=None)
	parser.add_argument("--bf16", type=int, default=0)
	parser.add_argument("--data_dir", type=str, default=None)
	parser.add_argument("--train_window_size", type=int, default=None)
	parser.add_argument("--num_processes_per_gpu", type=int, default=None)
	parser.add_argument("--start_exidx", type=int, default=None)
	parser.add_argument("--end_exidx", type=int, default=None)
	parser.add_argument("--num_epochs", type=int, default=None)
	parser.add_argument("--lr", type=float, default=None)
	parser.add_argument("--out_dir", type=str, default=None)
	parser.add_argument("--eval_window_size", type=int, default=None)
	args = parser.parse_args()

	queue = multiprocessing.Queue()  # queue used to load gpu ids
	n_gpu = len(args.gpu.split(','))
	num_processes_per_gpu = args.num_processes_per_gpu
	# initialize the queue with the GPU ids
	for _ in range(num_processes_per_gpu):
		for gpu_id in args.gpu.split(','):
			queue.put(gpu_id)
	
	assert args.bf16 in [0,1]
	args.bf16 = bool(args.bf16)
	mw = ModelWrapper(model_type='clm', model_name=args.model_name,
					  tokenizer=AutoTokenizer.from_pretrained(args.model_name),
					  load_pretrained=True,
					  load_pretrained_hf_name=args.model_name, bf16=args.bf16)
	data = load_from_disk(args.data_dir)
	manager = multiprocessing.Manager()
	start_exidx, end_exidx = args.start_exidx, args.end_exidx
	exidxs = range(start_exidx, min(end_exidx, len(data)))
	del data

	if not os.path.exists(args.out_dir):
		os.makedirs(args.out_dir, exist_ok=True)

	p = multiprocessing.Pool(processes=n_gpu * num_processes_per_gpu)
	run_exs = partial(detect_structure_batch_examples, mw=mw, data_dir=args.data_dir, # load data in each subprocess
					  train_window_size=args.train_window_size, eval_window_size=args.eval_window_size,
					  num_epochs=args.num_epochs, lr=args.lr, out_dir=args.out_dir)
	batch_size = 100
	batch_exidxs = [exidxs[ptr: ptr + batch_size]
					for ptr in range(0, len(exidxs), batch_size)]
	for _ in tqdm(p.imap(run_exs, batch_exidxs), total=len(batch_exidxs)):
		pass
	p.close()
	p.join()
	# merge all results
	exidx2out = {}
	for fname in os.listdir(args.out_dir):
		batch_exidx2out = pkl.load(open(f'{args.out_dir}/{fname}', 'rb'))
		if len(batch_exidx2out) == 0:
			exit(0)
		exidx2out.update(batch_exidx2out)
	pkl.dump(exidx2out, open(f'{args.out_dir}/exidx2out.pkl', 'wb'))
 