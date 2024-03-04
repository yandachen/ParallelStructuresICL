from accelerate import Accelerator
import os
from transformers import GPT2LMHeadModel, RobertaForMaskedLM, GPT2Config, RobertaConfig, AutoTokenizer, \
	get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup, GPTNeoXForCausalLM, GPTNeoXConfig, \
	OPTForCausalLM, OPTConfig
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import random
from torch.optim import AdamW
import math
import numpy as np
from datasets import load_from_disk, Dataset
import json
from torch.nn import CrossEntropyLoss
from scipy.special import softmax
import time
from copy import deepcopy
import pickle as pkl
import socket
import shutil
import torch
from typing import Optional, Tuple, Union
from flash_attn import __version__ as flash_attn_version
from flash_attn.bert_padding import pad_input, unpad_input
from flash_attn.flash_attn_interface import (
	flash_attn_func,
	flash_attn_varlen_kvpacked_func,
)
from transformers import GPT2LMHeadModel
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention, GPT2Block, GPT2Model

vanilla_attention = GPT2Attention.forward

import gc


def gpt2_flash_forward(
	self,
	hidden_states: Optional[Tuple[torch.FloatTensor]],
	layer_past: Optional[Tuple[torch.Tensor]] = None,
	attention_mask: Optional[torch.FloatTensor] = None,
	head_mask: Optional[torch.FloatTensor] = None,
	encoder_hidden_states: Optional[torch.Tensor] = None,
	encoder_attention_mask: Optional[torch.FloatTensor] = None,
	use_cache: Optional[bool] = False,
	output_attentions: Optional[bool] = False,
) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]], ...]:
	if encoder_hidden_states is not None:
		if not hasattr(self, "q_attn"):
			raise ValueError(
				"If class is used as cross attention, the weights `q_attn` have to be defined. "
				"Please make sure to instantiate class with `GPT2Attention(..., is_cross_attention=True)`."
			)
		query = self.q_attn(hidden_states)
		key, value = self.c_attn(encoder_hidden_states).split(
			self.split_size, dim=2)
		attention_mask = encoder_attention_mask
	else:
		query, key, value = self.c_attn(
			hidden_states).split(self.split_size, dim=2)
	query = self._split_heads(query, self.num_heads, self.head_dim)
	key = self._split_heads(key, self.num_heads, self.head_dim)
	value = self._split_heads(value, self.num_heads, self.head_dim)
	if layer_past is not None:
		past_key, past_value = layer_past
		key = torch.cat((past_key, key), dim=-2)
		value = torch.cat((past_value, value), dim=-2)
	if use_cache is True:
		present = (key, value)
	else:
		present = None
	if self.reorder_and_upcast_attn:
		attn_output, attn_weights = self._upcast_and_reordered_attn(
			query, key, value, attention_mask, head_mask)
	else:
		# attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask)
		query = query.transpose(1, 2)
		key = key.transpose(1, 2)
		value = value.transpose(1, 2)
		attn_weights = None
		if attention_mask is None:
			bsz = query.size(0)
			q_len = hidden_states.size(1)
			attn_output = flash_attn_func(query, key, value, 0.0, softmax_scale=None, causal=True).view(
				bsz, q_len, -1
			)
		else:
			bsz = query.size(0)
			q_len = hidden_states.size(1)
			attention_mask = ~(attention_mask < 0.0)[
				:, :, 0].view(bsz, -1).bool().clone()
			query, indices, cu_q_lens, max_s = unpad_input(
				query, attention_mask[:, -q_len:].clone())
			# We can skip concat and call unpad twice but seems better to call unpad only once.
			kv, _, cu_k_lens, max_k = unpad_input(
				torch.stack((key, value), dim=2), attention_mask
			)
			output_unpad = flash_attn_varlen_kvpacked_func(
				query,
				kv,
				cu_q_lens,
				cu_k_lens,
				max_s,
				max_k,
				0.0,
				softmax_scale=None,
				causal=True,
			)
			output_unpad = output_unpad.reshape(-1,
												self.num_heads * self.head_dim)
			attn_output = pad_input(output_unpad, indices, bsz, q_len)
	# attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
	attn_output = self.c_proj(attn_output)
	attn_output = self.resid_dropout(attn_output)
	outputs = (attn_output, present)
	if output_attentions:
		outputs += (attn_weights,)
	return outputs  # a, present, (attentions)


class ModelWrapper:
	def __init__(self, model_type, model_name, tokenizer, load_pretrained,
				 load_pretrained_hf_name=None, load_pretrained_hf_revision=None, load_pretrained_dir=None,
				 bf16=False):
		self.model_type = model_type
		self.model_name = model_name
		self.tokenizer = tokenizer
		assert self.model_type in ['mlm', 'clm']
		if 'gpt2' in self.model_name or 'pythia' in self.model_name:
			self.tokenizer.pad_token = self.tokenizer.eos_token
			self.tokenizer.padding_side = 'right'

		self.accelerator = None

		assert load_pretrained in [True, False]
		if load_pretrained:
			assert [load_pretrained_hf_name, load_pretrained_dir].count(
				None) == 1  # exactly one is not None
			if load_pretrained_dir is not None:
				self.load_pretrained_dir = load_pretrained_dir
		else:
			assert load_pretrained_hf_name is None and load_pretrained_dir is None

		self.bf16 = bf16
		if load_pretrained_hf_name is not None:
			if bf16 is False:
				torch_dtype = torch.float32
			else:
				torch_dtype = torch.bfloat16
			if 'gpt2' in self.model_name:
				self.model = GPT2LMHeadModel.from_pretrained(
					load_pretrained_hf_name, torch_dtype=torch_dtype)
			elif 'roberta' in self.model_name:
				self.model = RobertaForMaskedLM.from_pretrained(
					load_pretrained_hf_name, torch_dtype=torch_dtype)
			elif 'pythia' in self.model_name:
				self.model = GPTNeoXForCausalLM.from_pretrained(
					load_pretrained_hf_name, revision=load_pretrained_hf_revision, torch_dtype=torch_dtype)
			elif 'opt' in self.model_name:
				self.model = OPTForCausalLM.from_pretrained(
					load_pretrained_hf_name, torch_dtype=torch_dtype)
			else:
				raise NotImplementedError
		else:
			if 'gpt2' in self.model_name:
				self.model = GPT2LMHeadModel(
					config=GPT2Config.from_pretrained(self.model_name))
			elif 'roberta' in self.model_name:
				self.model = RobertaForMaskedLM(
					config=RobertaConfig.from_pretrained(self.model_name))
			elif 'pythia' in self.model_name:
				self.model = GPTNeoXForCausalLM(
					config=GPTNeoXConfig.from_pretrained(self.model_name))
			elif 'opt' in self.model_name:
				self.model = OPTForCausalLM(
					config=OPTConfig.from_pretrained(self.model_name))
			else:
				raise NotImplementedError
			if self.bf16:
				self.model.bfloat16()

			if not load_pretrained:  # loading a randomly initialized un-pretrained model
				# load initialized model state (use the same initialization for all experiments)
				if socket.gethostname() in ['tea', 'coffee']:
					model_init_path = f'/local/data/yanda_swordfish/ICLPre/initialized_models/{self.model_name}.pkl'
				else:
					model_init_path = f'/mnt/swordfish-datastore/yanda_swordfish/ICLPre/initialized_models/{self.model_name}.pkl'
				if not os.path.exists(model_init_path):
					print(
						'Re-run the code with python xxx.py to initialize the model with one python process only.')
					exit(1)
					# torch.save(self.model.state_dict(), model_init_path)
					# exit(1)
				assert os.path.exists(model_init_path)
				self.model.load_state_dict(torch.load(
					model_init_path, map_location='cpu'))

			elif load_pretrained_dir is not None:
				load_model_path = os.path.join(
					load_pretrained_dir, 'pytorch_model.bin')
				assert os.path.exists(load_model_path), load_model_path
				self.model.load_state_dict(torch.load(
					load_model_path, map_location='cpu'))

			else:
				raise NotImplementedError

	@staticmethod
	def _collate_fn_lm(batch_examples, pad_token_id, padding_side,
					   is_mlm=False, mlm_mask_token_id=None, mlm_vocab_size=None):
		"""
		mask tokens if the LM is a MLM
		otherwise batch and pad the input_ids
		batch_examples is List[Dict({'input_ids'})]
		"""
		lm_batch_examples = []
		for example in batch_examples:
			word_ids = example['input_ids']
			if not is_mlm:
				if 'labels' in example:
					lm_batch_examples.append({'input_ids': word_ids, 'labels': example['labels']})
				elif 'start_pred_pos' in example:
					start_pred_pos = example['start_pred_pos']
					labels = [-100] * start_pred_pos + word_ids[start_pred_pos:]
					assert len(word_ids) == len(labels)
					lm_batch_examples.append({'input_ids': word_ids, 'labels': labels})
				elif 'masked_tokenidxs' in example:
					labels = word_ids[:]
					for mask_tok_idx in example['masked_tokenidxs']:
						labels[mask_tok_idx] = -100
					lm_batch_examples.append({'input_ids': word_ids, 'labels': labels})
				else:
					lm_batch_examples.append({'input_ids': word_ids, 'labels': word_ids})
			else:
				predict_token_pos = random.sample(range(len(word_ids)), int(0.15 * len(word_ids)))
				input_ids = word_ids[:]
				labels = [-100] * len(word_ids)
				for pos in predict_token_pos:
					# only set labels for predict tokens
					labels[pos] = word_ids[pos]
					p = random.random()  # (0, 1)
					if p <= 0.8:
						input_ids[pos] = mlm_mask_token_id
					elif 0.8 < p <= 0.9:
						pass  # unchanged
					else:
						input_ids[pos] = random.sample(
							range(mlm_vocab_size), 1)[0]
				lm_batch_examples.append({'input_ids': input_ids, 'labels': labels})
		return ModelWrapper._collate_fn_pad(lm_batch_examples, pad_token_id=pad_token_id, padding_side=padding_side)

	@staticmethod
	def _collate_fn_pad(batch_examples, pad_token_id, padding_side):
		"""
		Batching and padding. Nothing else.
		batch_examples is a HF Dataset with feature "input_ids"
		For CLM, assume that the batch_examples in the parameter are the prompt.
		"""
		assert padding_side in ['left', 'right']
		num_examples = len(batch_examples)
		# pad to max_seq_length on the right hand side.
		max_seq_length = max([len(example['input_ids'])
							 for example in batch_examples])
		input_ids = torch.full((num_examples, max_seq_length), pad_token_id)
		attention_mask = torch.full((num_examples, max_seq_length), 0)
		position_ids = torch.full((num_examples, max_seq_length), 0)
		include_labels = 'labels' in batch_examples[0]
		if include_labels:
			labels = torch.full((num_examples, max_seq_length), -100)
		for ex_idx, example in enumerate(batch_examples):
			word_ids = example['input_ids']
			ex_len = len(word_ids)
			if padding_side == 'right':
				input_ids[ex_idx][: ex_len] = torch.LongTensor(word_ids)
				attention_mask[ex_idx][: ex_len] = 1
				position_ids[ex_idx][: ex_len] = torch.arange(ex_len)
				if include_labels:
					labels[ex_idx][: ex_len] = torch.LongTensor(example['labels'])
			elif padding_side == 'left':
				input_ids[ex_idx][-ex_len:] = torch.LongTensor(word_ids)
				attention_mask[ex_idx][-ex_len:] = 1
				position_ids[ex_idx][-ex_len:] = torch.arange(ex_len)
				if include_labels:
					labels[ex_idx][-ex_len:] = torch.LongTensor(example['labels'])
			else:
				raise NotImplementedError
		if not include_labels:
			return input_ids, attention_mask, position_ids
		else:
			return input_ids, attention_mask, position_ids, labels


	def evaluate_valdata_loss(self, val_dataloader):
		self.model.eval()
		dev_ex_losses = []
		for batch_data in val_dataloader:
			input_ids, attention_mask, position_ids, labels = batch_data
			with torch.no_grad():
				output = self.acce_model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, labels=labels)
				loss = output.loss
				batch_ex_losses = self.accelerator.gather_for_metrics(
					loss.repeat(input_ids.shape[0]))
			dev_ex_losses += batch_ex_losses.tolist()
		return np.mean(dev_ex_losses)

	def train(self, is_lm, train_data, val_data,
			  n_gpus, lr, per_device_batch_size, effective_bsz,
			  output_dir, save_model_ckpts, save_model_ckpts_steps=[],
			  num_epochs=None, num_train_steps=None, min_num_train_steps=None,
			  lr_scheduler=None, num_warmup_steps=None,
			  nonbias_weight_decay=0, shuffle_train=True,
			  gradient_checkpointing=False, use_flash_attention=False,
			  early_stopping=False,
			  #   early_stopping_window=None, early_stopping_diff=None, early_stopping_icl_max_seq_length=None
			  patience_num_train_steps=None, patience_perplexity_decrease=0.002
			  ):
		# train_data is a HF dict with the key "input_ids" and possibly "start_pred_pos"
		if is_lm:
			assert 'input_ids' in train_data.column_names
		else:
			assert train_data.column_names == ['input_ids', 'labels']
		if use_flash_attention:
			GPT2Attention.forward = gpt2_flash_forward
		if lr_scheduler is not None:
			assert num_warmup_steps is not None
		if num_warmup_steps is not None:
			assert lr_scheduler in ['cosine', 'linear']
		assert effective_bsz % (n_gpus * per_device_batch_size) == 0
		grad_accu_steps = effective_bsz // (n_gpus * per_device_batch_size)
		if gradient_checkpointing:
			self.model.gradient_checkpointing_enable()
		# either specify num_epochs or num_train_steps
		assert [num_epochs, num_train_steps].count(None) == 1
		# check early stopping parameters are specified
		if early_stopping:
			assert patience_num_train_steps is not None
			# assert [early_stopping_window, early_stopping_diff, early_stopping_icl_max_seq_length].count(None) == 0

		self.accelerator = Accelerator(
			gradient_accumulation_steps=grad_accu_steps)
		f_log = None
		if self.accelerator.is_main_process:
			if os.path.exists(output_dir):
				print('EXPERIMENT DIRECTOR ALREADY EXISTS:', output_dir)
				remove_dir = input(
					'Do you want to remove the existing directory?')
				if remove_dir == 'yes':
					shutil.rmtree(output_dir)
				else:
					exit(1)
			assert not os.path.exists(output_dir), output_dir
			os.makedirs(output_dir)
			log_fname = f'{output_dir}/log.txt'
			f_log = open(log_fname, 'a')
			f_log.write(f'#Training hyperparameters#\n')
			f_log.write(
				f'model_type: {self.model_type}, model_name: {self.model_name}\n'
				f'n_gpus: {n_gpus}, num_epochs: {num_epochs}, num_train_steps: {num_train_steps}, min_num_train_steps: {min_num_train_steps}, lr: {lr}, lr scheduler: {lr_scheduler}, nonbias_weight_decay: {nonbias_weight_decay}\n'
				f'per_device_bsz: {per_device_batch_size}, effective_bsz: {effective_bsz}, num_warmup_steps: {num_warmup_steps}\n'
				f'gradient_checkpointing: {gradient_checkpointing}, bf16: {self.bf16}, use_flash_attention: {use_flash_attention}\n'
				f'early_stopping: {early_stopping}, '
				#   early_stopping_window: {early_stopping_window}, early_stopping_diff: {early_stopping_diff}, early_stopping_icl_max_seq_length: {early_stopping_icl_max_seq_length}
				f'patience_num_train_steps: {patience_num_train_steps}, patience_perplexity_decrease: {patience_perplexity_decrease}\n')

		self.accelerator.print('loading data...')
		if self.accelerator.is_main_process:
			f_log.write(f'Number of training examples: {len(train_data)}\n')

		if is_lm and self.model_type == 'mlm':
			def collate_fn(batch_examples): return ModelWrapper._collate_fn_lm(batch_examples, self.tokenizer.pad_token_id, padding_side='right',
																			   is_mlm=True,
																			   mlm_mask_token_id=self.tokenizer.mask_token_id,
																			   mlm_vocab_size=self.tokenizer.vocab_size)
		elif is_lm and self.model_type == 'clm':
			def collate_fn(batch_examples): return ModelWrapper._collate_fn_lm(batch_examples, self.tokenizer.pad_token_id, padding_side='right',
																			   is_mlm=False)
		else:
			assert is_lm is False
			def collate_fn(batch_examples): return ModelWrapper._collate_fn_pad(
				batch_examples, self.tokenizer.pad_token_id, padding_side='right')
		train_dataloader = DataLoader(
			train_data, per_device_batch_size, shuffle=shuffle_train, collate_fn=collate_fn)
		val_dataloader = DataLoader(
			val_data, per_device_batch_size, shuffle=False, collate_fn=collate_fn)

		# set up optimizers
		bias_params = [params for param_name, params in self.model.named_parameters() if
					   'bias' in param_name or 'layer_norm.weight' in param_name]
		non_bias_params = [params for param_name, params in self.model.named_parameters() if
						   'bias' not in param_name and 'layer_norm.weight' not in param_name]
		self.accelerator.print(
			f'Number of non-bias parameters: {len(non_bias_params)}')
		self.accelerator.print(
			f'Number of bias parameters: {len(bias_params)}')

		if 'gpt2' in self.model_name:
			optimizer = AdamW([{'params': non_bias_params, 'lr': lr, 'weight_decay': 0.01},
							   {'params': bias_params, 'lr': lr, 'weight_decay': 0}])
		elif 'roberta' in self.model_name:
			optimizer = AdamW(
				# nonbias_weight_decay used to be 0.01
				[{'params': non_bias_params, 'lr': lr, 'weight_decay': nonbias_weight_decay, 'betas': (0.9, 0.999),
				  'eps': 1e-6},
				 {'params': bias_params, 'lr': lr, 'weight_decay': 0, 'betas': (0.9, 0.999), 'eps': 1e-6}])
		else:
			raise NotImplementedError

		# set up learning rate scheduler
		if num_epochs is not None:
			num_training_batches = len(train_dataloader) * num_epochs
		else:
			assert num_train_steps is not None
			num_training_batches = num_train_steps * \
				(effective_bsz // per_device_batch_size)
		if num_warmup_steps is not None:
			if type(num_warmup_steps) == int:
				# assert num_warmup_steps > 100
				num_warmup_steps = num_warmup_steps * grad_accu_steps
			# When performing gradient accumulation scheduler lengths should not be changed accordingly, accelerate will always  step the scheduler to account for it.
			# accelerate/src/accelerate/scheduler.py Lines 31 to 32 in b0f8189
			elif type(num_warmup_steps) == float:
				assert 0 <= num_warmup_steps <= 1
				num_warmup_steps = num_warmup_steps * num_training_batches

		if lr_scheduler is not None:
			get_lr_scheduler = \
				{'cosine': get_cosine_schedule_with_warmup,
					'linear': get_linear_schedule_with_warmup}[lr_scheduler]
			lr_scheduler = get_lr_scheduler(optimizer,
											num_warmup_steps=num_warmup_steps,
											# adapted from line 494 of here: https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_clm_no_trainer.py
											# https://github.com/huggingface/accelerate/blob/main/examples/by_feature/checkpointing.py#L152-L164
											# https://github.com/huggingface/accelerate/issues/628
											num_training_steps=num_training_batches)  # accelerate will handle gradient accumulation for us

		if self.accelerator.is_main_process:
			f_log.write(
				f'per device batch size: {per_device_batch_size}, gradient accumulation steps: {grad_accu_steps}, effective batch size: {per_device_batch_size * n_gpus * grad_accu_steps}' + '\n')
			f_log.flush()

		if self.accelerator.is_main_process:
			f_log.write(
				f'Size of train dataloader before accelerator.prepare: {len(train_dataloader)}\n')
			f_log.flush()

		if lr_scheduler is None:
			train_dataloader, val_dataloader, self.acce_model, optimizer = self.accelerator.prepare(
				train_dataloader, val_dataloader, self.model, optimizer)
		else:
			train_dataloader, val_dataloader, self.acce_model, optimizer, lr_scheduler = self.accelerator.prepare(
				train_dataloader, val_dataloader, self.model, optimizer, lr_scheduler)

		if self.accelerator.is_main_process:
			f_log.write(
				f'Size of train dataloader after accelerator.prepare: {len(train_dataloader)}\n')
			f_log.flush()

		# The actual batch size for your training will be the number of devices used multiplied by the batch size you set in your script
		# cf: https://huggingface.co/docs/accelerate/quicktour
		# Your training dataloader may change length when going through this method: if you run on X GPUs, it will have its length divided by X (since your actual batch size will be multiplied by X)

		# need to recalculate here as the length of train_dataloader may have changed
		if num_epochs is not None:
			num_update_steps_per_epoch = math.ceil(
				len(train_dataloader) / grad_accu_steps)
			progress_bar = tqdm(range(num_epochs * num_update_steps_per_epoch),
								desc='step')  # line 508 of https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_clm_no_trainer.py
		else:
			assert num_train_steps is not None
			progress_bar = tqdm(range(num_train_steps), desc='step')
			num_update_steps_per_epoch = math.ceil(
				len(train_dataloader) / grad_accu_steps)
			num_epochs = math.ceil(
				num_train_steps / num_update_steps_per_epoch)

		# TRAIN
		self.acce_model.train()
		start_time = time.time()
		train_losses = []
		completed_steps = 0
		if early_stopping:
			optimal_completed_steps = 0
			optimal_val_loss = np.inf
			step2val_loss = {}
			# early_stopping_accs = []
		end_training = False
		for _ in range(num_epochs):
			for batch_idx, batch_data in enumerate(train_dataloader):
				# print(len(batch_data[0]))
				# print(batch_idx)
				if self.accelerator.sync_gradients:
					# print('sync>>')
					if completed_steps in save_model_ckpts_steps:
						self.accelerator.wait_for_everyone()
						if self.accelerator.is_main_process and save_model_ckpts:
							self.accelerator.save_state(f'{output_dir}/step{completed_steps}.ckpt')
						elif self.accelerator.is_main_process and (not save_model_ckpts):
							os.makedirs(f'{output_dir}/step{completed_steps}.ckpt', exist_ok=True) # create an empty ckpt directory to save loss files
						self.accelerator.wait_for_everyone()
						# logging
						if len(train_losses) > 0:
							train_loss = np.mean(train_losses)
						else:
							train_loss = np.nan
						ckpt_time = time.time()
						training_time = (ckpt_time - start_time) // 60
						train_losses = []
						# calculate dev perplexity if early_stopping
						if early_stopping:
							# GPT2Attention.forward = vanilla_attention
							val_loss = self.evaluate_valdata_loss(val_dataloader)
							step2val_loss[completed_steps] = val_loss
							pkl.dump(val_loss, open(
								f'{output_dir}/step{completed_steps}.ckpt/dev_loss.pkl', 'wb'))
							loss_message = f'step {completed_steps}: train loss {train_loss:.4f}, val loss {val_loss} ({training_time} min)'
							if val_loss < optimal_val_loss:
								optimal_completed_steps = completed_steps
								optimal_val_loss = val_loss
						else:
							loss_message = f'step {completed_steps}: train loss {train_loss:.4f} ({training_time} min)'
						self.accelerator.print(loss_message)
						if self.accelerator.is_main_process:
							f_log.write(loss_message + '\n')
							f_log.flush()
						# early stopping based on score convergence
						# if early_stopping and (completed_steps >= min_num_train_steps) and (len(early_stopping_accs) >= early_stopping_window) and \
						# 	(np.max(early_stopping_accs[-early_stopping_window:]) - np.min(early_stopping_accs[-early_stopping_window:]) < early_stopping_diff):
							# abs(early_stopping_accs[-1] - np.mean(early_stopping_accs[-1-early_stopping_window:-1])) < early_stopping_diff:
						if early_stopping and (completed_steps >= min_num_train_steps):
							# check if min val_loss before (0, completed_steps - patience_num_train_steps) is > val_loss in (completed_steps - patience_num_train_steps, completed_steps) + epsilon
							early_vallosses = [step2val_loss[step] for step in step2val_loss if step <= completed_steps - patience_num_train_steps]
							recent_vallosses = [step2val_loss[step] for step in step2val_loss if step > completed_steps - patience_num_train_steps]
							if (len(early_vallosses)) > 0 and (np.min(early_vallosses) < np.min(recent_vallosses) + patience_perplexity_decrease):
								# and (completed_steps - optimal_completed_steps > patience_num_train_steps):
								end_training = True
								# break
						self.acce_model.train()
					if (num_train_steps is not None) and (completed_steps >= num_train_steps):
						end_training = True
					# if end_training:
					# 	break
				if end_training:
					continue
				with self.accelerator.accumulate(self.acce_model):
					input_ids, attention_mask, position_ids, labels = batch_data
					# model forward and backward
					output = self.acce_model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, labels=labels)
					loss = output.loss
					self.accelerator.backward(loss)
					optimizer.step()
					if lr_scheduler is not None:
						lr_scheduler.step()
					optimizer.zero_grad()
					train_losses.append(loss.item())
				if self.accelerator.sync_gradients:
					progress_bar.update(1)
					completed_steps += 1
			if end_training:
				break

		# save checkpoint
		self.accelerator.wait_for_everyone()
		if self.accelerator.is_main_process and save_model_ckpts:
			if not os.path.exists(f'{output_dir}/step{completed_steps}.ckpt'):
				self.accelerator.save_state(
					f'{output_dir}/step{completed_steps}.ckpt')
				# scores, avg_score = self.evaluate_icl(max_seq_length=early_stopping_icl_max_seq_length)
				# if self.bf16:
				# 	pkl.dump(scores, open(f'{output_dir}/step{completed_steps}.ckpt/icl-scores-bf16.pkl', 'wb'))
				# else:
				# 	pkl.dump(scores, open(f'{output_dir}/step{completed_steps}.ckpt/icl-scores.pkl', 'wb'))
		self.accelerator.wait_for_everyone()
		if self.accelerator.is_main_process:
			f_log.close()
		if use_flash_attention:  # revert to norm attention, only use flash attention for training
			GPT2Attention.forward = vanilla_attention
		# copy the model checkpoint as the final checkpoint
		if self.accelerator.is_main_process:
			shutil.copytree(
				f'{output_dir}/step{optimal_completed_steps}.ckpt', f'{output_dir}/model.ckpt')
		return optimal_val_loss

	def train_one_example(self, train_lm, train_data, val_data, num_epochs, lr, patience, eval_window_size, eval_stride, use_flash_attention=False, debug=False):
		if use_flash_attention:
			GPT2Attention.forward = gpt2_flash_forward
		if train_lm is False:
			collate_fn = ModelWrapper._collate_fn_pad
		else:
			collate_fn = ModelWrapper._collate_fn_lm
		# no need to use a dataloader; only one batch
		train_data = collate_fn(train_data, self.tokenizer.pad_token_id, 'right')
		# no need to use a dataloader; only one batch
		val_data_dict = val_data
		val_data = collate_fn(val_data, self.tokenizer.pad_token_id, 'right')
		bias_params = [params for param_name, params in self.model.named_parameters() if
					   'bias' in param_name.split('.')]
		non_bias_params = [params for param_name, params in self.model.named_parameters() if
						   'bias' not in param_name.split('.')]
		optimizer = None
		if 'gpt2' in self.model_name:
			optimizer = AdamW([{'params': non_bias_params, 'lr': lr, 'weight_decay': 0.01},
							   {'params': bias_params, 'lr': lr, 'weight_decay': 0}])
		elif 'roberta' in self.model_name:
			optimizer = AdamW(
				[{'params': non_bias_params, 'lr': lr, 'weight_decay': 0.01, 'betas': (0.9, 0.999), 'eps': 1e-6},
				 {'params': bias_params, 'lr': lr, 'weight_decay': 0, 'betas': (0.9, 0.999), 'eps': 1e-6}])
		self.model.cuda()
		optimal_epoch_idx, optimal_model = None, None
		ckpts_tokenlosses = []
		val_masked_tokenidxs = val_data_dict['masked_tokenidxs'][0]
		val_unmasked_tokenidxs = [tokenidx for tokenidx in range(len(val_data[0][0])) if tokenidx not in val_masked_tokenidxs] # val_data[0] is input_ids
		for epoch_idx in range(num_epochs):
			self.model.train()
			input_ids, attention_mask, position_ids, labels = train_data
			# model forward and backward
			output = self.model(input_ids=input_ids.cuda(), attention_mask=attention_mask.cuda(),
								position_ids=position_ids.cuda(), labels=labels.cuda())
			train_loss = output.loss
			train_loss.backward()
			optimizer.step()
			optimizer.zero_grad()
			self.model.eval()
			input_ids, attention_mask, position_ids, labels = val_data
			with torch.no_grad():
				# output = self.model(input_ids=input_ids.cuda(), attention_mask=attention_mask.cuda(),
				# 					position_ids=position_ids.cuda(), labels=labels.cuda())
				# val_loss = output.loss
				token_losses = self.evaluate_perplexity_limit_context_size(eval_data=val_data_dict, per_device_batch_size=256, window_size=eval_window_size, stride=eval_stride, use_accelerator=False)['token_losses'][0]
				# assert abs(np.nanmean(np.array(token_losses)[val_unmasked_tokenidxs]) - val_loss) < 1e-3 # checked
				assert not np.any(np.isnan(np.array(token_losses)[val_unmasked_tokenidxs]))
				ckpts_tokenlosses.append(token_losses)
			# early stopping when token losses stop decrease on all eval_id tokens that are not masked
			unmasked_tokenlosses = np.array(ckpts_tokenlosses)[:, val_unmasked_tokenidxs]
			assert not np.any(np.isnan(unmasked_tokenlosses))
			assert unmasked_tokenlosses.shape[1] == len(val_unmasked_tokenidxs)
			if (epoch_idx == 0) or (np.any(unmasked_tokenlosses[-1] < np.max(unmasked_tokenlosses[:-1], axis=0))):
				optimal_epoch_idx = epoch_idx
			# if val_loss < optimal_val_loss:
			# 	optimal_val_loss = val_loss
			# 	optimal_epoch_idx = epoch_idx
			# 	optimal_model = deepcopy(self.model)
			if epoch_idx - optimal_epoch_idx > patience:
				break
		self.model = optimal_model
		return ckpts_tokenlosses

	def evaluate_perplexity(self, eval_data, per_device_batch_size, use_accelerator=True, show_progress_bar=False):
		# eval_data is a HF Dataset
		assert 'input_ids' in eval_data.column_names
		if use_accelerator and self.accelerator is None:
			self.accelerator = Accelerator()
		if self.model_type == 'mlm':
			eval_dataloader = DataLoader(eval_data, per_device_batch_size, shuffle=False,
										 collate_fn=lambda batch_examples:
										 ModelWrapper._collate_fn_lm(batch_examples, self.tokenizer.pad_token_id, padding_side='right',
																	 is_mlm=True,
																	 mlm_mask_token_id=self.tokenizer.mask_token_id,
																	 mlm_vocab_size=self.tokenizer.vocab_size))
		elif self.model_type == 'clm':
			eval_dataloader = DataLoader(eval_data, per_device_batch_size, shuffle=False,
										 collate_fn=lambda batch_examples:
										 ModelWrapper._collate_fn_lm(batch_examples, self.tokenizer.pad_token_id, padding_side='right',
																	 is_mlm=False))
		else:
			raise NotImplementedError

		if use_accelerator:
			eval_dataloader = self.accelerator.prepare(eval_dataloader)
			if hasattr(self, 'acce_model') is False:
				self.acce_model = self.accelerator.prepare(self.model)
		else:
			self.acce_model = self.model
			self.acce_model.cuda()
		self.acce_model.eval()
		# adapted from here: https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_clm_no_trainer.py
		eval_token_losses = []
		loss_fct = CrossEntropyLoss(reduction='none')
		if show_progress_bar:
			pbar = tqdm(total=len(eval_dataloader))
		for batch_data in eval_dataloader:
			input_ids, attention_mask, position_ids, labels = batch_data
			with torch.no_grad():
				if not use_accelerator:  # manually send all tensors to cuda
					input_ids, attention_mask, position_ids, labels = \
						input_ids.cuda(), attention_mask.cuda(), position_ids.cuda(), labels.cuda()
				output = self.acce_model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, labels=labels)
			avg_ex_loss = output.loss  # batch loss
			# calculate loss at each token for each example, be careful about how padding changes abs pos
			batch_token_losses = []
			batch_num_tokens = []
			for ex_idx in range(len(input_ids)):
				if self.model_type == 'clm':
					token_losses = loss_fct(output.logits[ex_idx][: -1], labels[ex_idx][1:]).tolist()
					# no loss predicted for the index-0 token
					token_losses = [np.nan] + token_losses
				elif self.model_type == 'mlm':
					token_losses = loss_fct(output.logits[ex_idx], labels[ex_idx]).tolist()
				else:
					raise NotImplementedError
				# token_losses[idx] is loss of predicting token[idx]
				assert len(token_losses) == len(input_ids[ex_idx])
				# for those positions where labels=-100 should set token_losses to be nan
				token_losses = torch.tensor(token_losses)
				token_losses[labels[ex_idx] == -100] = np.nan
				num_tokens = torch.sum(input_ids[ex_idx] != self.tokenizer.pad_token_id)
				batch_token_losses.append(token_losses)
				batch_num_tokens.append(num_tokens)
			batch_token_losses = torch.stack(batch_token_losses).to(input_ids.get_device())
			batch_num_tokens = torch.tensor(batch_num_tokens).to(input_ids.get_device())
			if use_accelerator:
				batch_token_losses = self.accelerator.gather_for_metrics(batch_token_losses)
				batch_num_tokens = self.accelerator.gather_for_metrics(batch_num_tokens)
			assert len(batch_token_losses) == len(batch_num_tokens)
			# sanity check that the average of (non-zero and non-nan) batch_token_losses is equal to avg_ex_loss
			if (self.bf16 is False) and (use_accelerator is False):
				assert torch.abs(torch.nanmean(batch_token_losses) - avg_ex_loss) < 1e-3, (torch.nanmean(batch_token_losses) - avg_ex_loss)
			nopad_token_losses = []
			for token_losses, num_tokens in zip(batch_token_losses, batch_num_tokens):
				nopad_token_losses.append(token_losses[:num_tokens].cpu().tolist())
			eval_token_losses += nopad_token_losses
			if show_progress_bar:
				pbar.update(1)
		assert len(eval_token_losses) == len(eval_data)
		for data, token_losses in zip(eval_data, eval_token_losses):
			assert len(data['input_ids']) == len(token_losses)
		dev_loss = np.nanmean([loss for token_losses in eval_token_losses for loss in token_losses])
		perplexity = np.exp(dev_loss)
		return {'loss': dev_loss, 'perplexity': perplexity, 'token_losses': eval_token_losses}

	def evaluate_perplexity_limit_context_size(self, eval_data, window_size, stride, per_device_batch_size, use_accelerator=True, show_progress_bar=False):
		# eval_data is a HF Dataset
		assert 'input_ids' in eval_data.column_names
		assert window_size >= stride + 1 # 1 is needed because the first token of each window is predicted loss nan
		# segment into windows
		eval_data_windows, window_exidx_offset_targetstartpos = [], []
		for exidx in range(len(eval_data)):
			input_ids = eval_data[exidx]['input_ids']
			eval_data_windows.append(input_ids[: window_size])
			window_exidx_offset_targetstartpos.append((exidx, 0, 0))
			for offset in range(stride, len(input_ids) - (window_size - stride), stride):
				window = input_ids[offset: offset + window_size]
				target_start_pos = window_size - stride
				if len(window[target_start_pos:]) <= 0:
					break
				eval_data_windows.append(window)
				window_exidx_offset_targetstartpos.append((exidx, offset, target_start_pos))
		eval_data_windows = Dataset.from_dict({'input_ids': eval_data_windows})
		loss = self.evaluate_perplexity(eval_data_windows, per_device_batch_size=per_device_batch_size, use_accelerator=use_accelerator, show_progress_bar=show_progress_bar)
		window_token_losses = loss['token_losses']
		assert len(window_token_losses) == len(eval_data_windows) == len(window_exidx_offset_targetstartpos)
		# create an empty matrix of token losses to fill in
		orig_token_losses = [np.full(len(ex['input_ids']), np.nan) for ex in eval_data]
		for token_losses, exidx_offset_targetstartpos in zip(window_token_losses, window_exidx_offset_targetstartpos):
			exidx, offset, target_start_pos = exidx_offset_targetstartpos
			target_len = len(token_losses[target_start_pos:])
			assert np.all(np.isnan(orig_token_losses[exidx][offset + target_start_pos: offset + target_start_pos + target_len])) # the predicted targets of different windows do not overlap
			orig_token_losses[exidx][offset + target_start_pos: offset + target_start_pos + target_len] = token_losses[target_start_pos: ]
		# check that all tokens except token 0 are covered in the target of at least one window
		for exidx in range(len(orig_token_losses)):
			assert not np.any(np.isnan(orig_token_losses[exidx][1:]))
		orig_token_losses = [token_losses.tolist() for token_losses in orig_token_losses]
		assert len(orig_token_losses) == len(eval_data)
		dev_loss = np.nanmean([loss for token_losses in orig_token_losses for loss in token_losses])
		perplexity = np.exp(dev_loss)
		return {'loss': dev_loss, 'perplexity': perplexity, 'token_losses': orig_token_losses}

	def generate_one_token(self, eval_data, per_device_batch_size, options=None):
		"""
		eval_data is a HF Dataset with feature "input_ids"
		For MLM, assume that the batch_examples in the parameter already contains one <mask>.
		options is either None, or a list of word ids.
		"""
		assert eval_data.column_names == ['input_ids']
		if self.accelerator is None:
			self.accelerator = Accelerator()
		eval_dataloader = DataLoader(eval_data, per_device_batch_size, shuffle=False,
									 collate_fn=lambda batch_examples:
									 ModelWrapper._collate_fn_pad(batch_examples, self.tokenizer.pad_token_id, padding_side='left'))

		# sanity check that if MLM then each example of eval_data has exactly one mask_token_id
		if self.model_type == 'mlm':
			for ex in eval_data:
				assert ex['input_ids'].count(self.tokenizer.mask_token_id) == 1

		eval_dataloader = self.accelerator.prepare(eval_dataloader)
		if hasattr(self, 'acce_model') is False:
			self.acce_model = self.accelerator.prepare(self.model)
		self.acce_model.eval()

		token_preds = []
		if type(options) is list:
			options = torch.tensor(options)
		for batch_data in eval_dataloader:
			input_ids, attention_mask, position_ids = batch_data
			output = self.acce_model.forward(input_ids=input_ids, attention_mask=attention_mask,
											 position_ids=position_ids, return_dict=True)
			target_token_logits = None
			if self.model_type == 'clm':
				target_token_logits = output.logits[:, -1, :]
			elif self.model_type == 'mlm':
				target_token_logits = []
				for ex_idx in range(input_ids.shape[0]):
					mask_token_idxs = (
						input_ids[ex_idx] == self.tokenizer.mask_token_id).nonzero().flatten()
					# only one <mask> token in each example
					assert len(mask_token_idxs) == 1
					target_token_logits.append(
						output.logits[ex_idx][mask_token_idxs[0]])
				target_token_logits = torch.vstack(target_token_logits)
			assert target_token_logits.shape[0] == input_ids.shape[0]
			assert target_token_logits.shape[1] == output.logits.shape[-1]
			if options is not None:
				target_token_logits = target_token_logits[:, options]
				assert target_token_logits.shape[0] == input_ids.shape[0]
				assert target_token_logits.shape[1] == len(options)
			preds = torch.argmax(target_token_logits, dim=1)
			assert len(preds) == input_ids.shape[0]
			if options is not None:
				options = options.to(input_ids.get_device())
				preds = options[preds]
			all_preds = self.accelerator.gather_for_metrics(preds)
			token_preds += all_preds.cpu().tolist()
		assert len(token_preds) == len(eval_data)
		return token_preds

	def score_options(self, data, max_seq_length, per_device_batch_size, return_normalized_probs=True):
		"""
		eval_data is a HF Dataset with two features 'prompt_input_ids', 'options_input_ids'
		"""
		assert data.column_names == ['prompt_input_ids', 'options_input_ids']
		if self.accelerator is None:
			self.accelerator = Accelerator()

		input_ids, labels = [], []
		for example in data:
			prompt_input_ids, options_input_ids = example['prompt_input_ids'], example['options_input_ids']
			if self.model_type == 'clm':
				for option_input_ids in options_input_ids:
					input_ids.append(prompt_input_ids + option_input_ids)
					labels.append(
						[-100] * len(prompt_input_ids) + option_input_ids)
			elif self.model_type == 'mlm':
				for option_input_ids in options_input_ids:
					input_ids.append(
						prompt_input_ids + [self.tokenizer.mask_token_id] * len(option_input_ids))
					labels.append(
						[-100] * len(prompt_input_ids) + option_input_ids)

		input_ids = [ex[-max_seq_length:] for ex in input_ids]
		labels = [ex[-max_seq_length:] for ex in labels]
		eval_data = Dataset.from_dict(
			{'input_ids': input_ids, 'labels': labels})
		eval_dataloader = DataLoader(eval_data, per_device_batch_size, shuffle=False,
									 collate_fn=lambda batch_examples:
									 ModelWrapper._collate_fn_pad(batch_examples, self.tokenizer.pad_token_id, padding_side='right'))
		self.model.eval()
		# here do not override self.model so that the same model wrapper can be used multiple times to save loading time.
		eval_dataloader = self.accelerator.prepare(eval_dataloader)
		if hasattr(self, 'acce_model') is False:
			self.acce_model = self.accelerator.prepare(self.model)
		self.acce_model.eval()

		eval_data_losses = []
		# no length normalization is applied.
		loss_fct = CrossEntropyLoss(reduction='sum')
		for batch_data in eval_dataloader:
			input_ids, attention_mask, position_ids, labels = batch_data
			with torch.no_grad():
				output = self.acce_model.forward(input_ids=input_ids, attention_mask=attention_mask,
												 position_ids=position_ids)
				batch_losses = []
				for ex_idx in range(len(input_ids)):
					if self.model_type == 'clm':  # note: if using celoss to calculate loss manually for CLMs, remember to shift the logits and labels!
						ex_loss = loss_fct(
							output.logits[ex_idx][: -1], labels[ex_idx][1:])
					elif self.model_type == 'mlm':
						ex_loss = loss_fct(
							output.logits[ex_idx], labels[ex_idx])
					else:
						raise NotImplementedError
					batch_losses.append(ex_loss)
				batch_losses = torch.stack(batch_losses)
			all_losses = self.accelerator.gather_for_metrics(
				batch_losses)  # Remember that gather_for_metrics has to take Tensors on CUDA as input
			eval_data_losses += all_losses.cpu().tolist()

		assert len(eval_data_losses) == len(
			eval_data), (len(eval_data_losses), len(eval_data))

		pred_probs = []
		cur_ptr = 0
		for ex_idx in range(len(data)):
			num_options = len(data[ex_idx]['options_input_ids'])
			ex_pred_loss = eval_data_losses[cur_ptr: cur_ptr + num_options]
			if return_normalized_probs:
				pred_probs.append(softmax(-np.array(ex_pred_loss)))
			else:
				pred_probs.append(np.exp(-np.array(ex_pred_loss)))
			cur_ptr += num_options
		assert cur_ptr == len(eval_data_losses)
		assert len(pred_probs) == len(data)
		return np.array(pred_probs)

	def generate(self, eval_data, per_device_batch_size, max_new_tokens, eos_token_id=None):
		"""
		eval_data is a HF Dataset with feature "input_ids"
		stop_token_ids is an id that will end the generation
		max_new_tokens is the maximum number of new tokens
		"""
		assert self.model_type == 'clm'  # only CLM can generate autoregressively
		assert eval_data.column_names == ['input_ids']
		self.model.eval()
		if self.accelerator is None:
			self.accelerator = Accelerator()
		eval_dataloader = DataLoader(eval_data, per_device_batch_size, shuffle=False,
									 collate_fn=lambda batch_examples:
									 ModelWrapper._collate_fn_pad(batch_examples, self.tokenizer.pad_token_id, padding_side='left'))
		eval_dataloader = self.accelerator.prepare(eval_dataloader)
		if hasattr(self, 'acce_model') is False:
			self.acce_model = self.accelerator.prepare(self.model)
		self.acce_model.eval()

		all_padded_outputs = []
		for batch_data in eval_dataloader:
			input_ids, attention_mask, position_ids = batch_data
			# 'DistributedDataParallel' object has no attribute 'generate', so need to unwrap first
			# cf: transformers/examples/pytorch/summarization/run_summarization_no_trainer.py, line 675
			outputs = self.accelerator.unwrap_model(self.acce_model).generate(input_ids=input_ids,
																			  attention_mask=attention_mask,
																			  pad_token_id=self.tokenizer.eos_token_id,
																			  max_new_tokens=max_new_tokens,
																			  eos_token_id=eos_token_id,
																			  do_sample=False, num_beams=1)
			assert torch.all(input_ids == outputs[:, : input_ids.shape[1]])
			outputs = outputs[:, input_ids.shape[1]:]
			padded_outputs = self.accelerator.pad_across_processes(outputs, dim=1, pad_first=False,  # pad on the right
																   pad_index=self.tokenizer.pad_token_id)
			# padded_outputs = torch.hstack([torch.full((len(input_ids), max_new_tokens - outputs.shape[1]),
			# 										  self.tokenizer.pad_token_id).cuda(),
			# 							   outputs])
			padded_outputs = self.accelerator.gather_for_metrics(
				padded_outputs.float())
			all_padded_outputs += padded_outputs
		assert len(all_padded_outputs) == len(eval_data)
		# remove the padding and stop at the eos token
		outputs = []
		for padded_output in all_padded_outputs:
			padded_output = padded_output.int().tolist()
			# remove the padding on the LHS
			first_nonpad_token_idx = 0
			while first_nonpad_token_idx < len(padded_output) and padded_output[first_nonpad_token_idx] == self.tokenizer.pad_token_id:
				first_nonpad_token_idx += 1
			output = padded_output[first_nonpad_token_idx:]
			if eos_token_id not in output:
				outputs.append(output)
			else:
				outputs.append(output[: output.index(eos_token_id)])
		assert len(outputs) == len(eval_data)
		return outputs

