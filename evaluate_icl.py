import random
from datasets import Dataset
import numpy as np
from sklearn.metrics import accuracy_score
from transformers import AutoTokenizer
import json
from tqdm import tqdm, trange
import pickle as pkl


def icl(mw, examples, k, per_device_batch_size, max_seq_length, seed):
    random.seed(seed)
    num_examples = len(examples)
    # randomly sample special token ids
    token_ids = range(5, 50000)
    specialtokens = ['INPUT_ARG_DIVIDER', 'INPUT_OUTPUT_DEL', 'EX_EOS', 'yes', 'no'] + [f'number_{num}' for num in range(20)]
    specialtoken2wordid_list = []
    for exidx in range(len(examples)):
        specialtokenids = random.sample(token_ids, len(specialtokens))
        specialtoken2wordid_list.append({token: tokenid for token, tokenid in zip(specialtokens, specialtokenids)})
    assert type(specialtoken2wordid_list) == list and len(specialtoken2wordid_list) == num_examples
    prompts = []
    for exidx in range(num_examples):
        # sample few-shot examples
        dems = random.sample(examples[: exidx] + examples[exidx+1:], k)
        query = examples[exidx]
        prompt = []
        if specialtoken2wordid_list is None:
            for dem in dems:
                prompt += dem['input'] + dem['output']
            prompt += query['input']
        else:
            for dem in dems:
                prompt += replace_special_token_with_wordid(dem['input'], specialtoken2wordid_list[exidx]) + \
                    replace_special_token_with_wordid(dem['output'], specialtoken2wordid_list[exidx])
            prompt += replace_special_token_with_wordid(query['input'], specialtoken2wordid_list[exidx])
        prompts.append(prompt)
    # classification or generation.
    score_options = 'options' in examples[0]
    if score_options:
        options = [[replace_special_token_with_wordid(option, specialtoken2wordid_list[exidx]) for option in ex['options']] for exidx, ex in enumerate(examples)]
        data = Dataset.from_dict({'prompt_input_ids': prompts, 'options_input_ids': options})
        pred_probs = mw.score_options(data, max_seq_length=max_seq_length, per_device_batch_size=per_device_batch_size, return_normalized_probs=False)
        assert pred_probs.shape[0] == len(examples)
        gts = [ex['options'].index(ex['output']) for ex in examples]
        assert -1 not in gts
        preds = np.argmax(pred_probs, axis=1)
        acc = accuracy_score(gts, preds)
        return acc
    else:
        # cut if exceeds maximum length
        max_output_length = max([len(ex['output']) for ex in examples])
        prompts = [prompt[-(max_seq_length - max_output_length):] for prompt in prompts]
        data = Dataset.from_dict({'input_ids': prompts})
        outputs = mw.generate(data, per_device_batch_size, max_new_tokens=max_output_length)
        assert len(outputs) == len(examples)
        # cut the outputs at EX_EOS and revert the special tokens in the outputs from word ids to special tokens
        truncated_reverted_outputs = []
        for exidx in range(len(examples)):
            output = outputs[exidx]
            specialtoken2wordid = specialtoken2wordid_list[exidx]
            wordid2specialtoken = {specialtoken2wordid[token]: token for token in specialtoken2wordid}
            # truncate at EX_EOS if EX_EOS appears in the model output
            eos_token_id = specialtoken2wordid['EX_EOS']
            if eos_token_id in output:
                # do not keep the EX_EOS token
                output = output[: output.index(eos_token_id)]
            # revert the special tokens from word ids to strings
            reverted_output = [wordid if wordid not in wordid2specialtoken else wordid2specialtoken[wordid] for wordid in output]
            truncated_reverted_outputs.append(reverted_output)
        assert len(truncated_reverted_outputs) == len(examples)
        # [:-1] strips the EX_EOS at the end.
        accs = [ex['output'][:-1] == out for ex, out in zip(examples, truncated_reverted_outputs)]
        acc = np.mean(accs)
        return acc


def replace_special_token_with_wordid(words, special_token2wordid):
    output = []
    for word in words:
        if type(word) == int:
            output.append(word)
        else:
            assert word in special_token2wordid
            output.append(special_token2wordid[word])
    return output


def icl_main(mw, max_seq_length, bsz, score_fname):
    task2k2accs = {}
    task2data = json.load(open('./icl_eval_data.json'))
    for task in tqdm(task2data, desc='task'):
        for k in [64, 96, 128]:
            examples = task2data[task]
            num_seeds = min(5, int(4000 / len(examples)))
            seeds = [42 * x for x in range(1, num_seeds+1)]
            accs = [icl(mw, examples, k, bsz, max_seq_length, seed) for seed in seeds]
            if task not in task2k2accs:
                task2k2accs[task] = {}
            task2k2accs[task][k] = accs
    pkl.dump(task2k2accs, open(score_fname, 'wb'))

