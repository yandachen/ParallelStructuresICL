from copy import deepcopy
from multiprocessing import Pool
import pickle as pkl
import random
import numpy as np
from tqdm import tqdm, trange
from collections import Counter
from datasets import load_from_disk, Dataset
import os


class QuickSelector:
    def __init__(self):
        pass

    def findKthLargest(self, nums, k):
        self.nums=nums
        self.k=k
        return self.quickselect(0,len(nums)-1)
        
    def partition(self,start,end):
        pivot_ind=random.randint(start,end)
        pivot=self.nums[pivot_ind]
        self.nums[pivot_ind],self.nums[end]=self.nums[end],self.nums[pivot_ind]
        pindex=start
        for i in range(start,end+1):
            if self.nums[i]>pivot:
                self.nums[i],self.nums[pindex]=self.nums[pindex],self.nums[i]
                pindex+=1
        self.nums[end],self.nums[pindex]=self.nums[pindex],self.nums[end]
        return pindex
        
    def quickselect(self,start,end):
        k=self.k -1
        if start<=end:
            pindex=self.partition(start,end)
            if pindex>k:
                return self.quickselect(start,pindex-1)
            elif pindex<k:
                return self.quickselect(pindex+1,end)
            else:
                return self.nums[k]
            

class UnigramSampler():
    def __init__(self, data, sample_distr):
        assert sample_distr in ['uniform', 'unigram']
        tokens = []
        for ex in data:
            tokens += ex['input_ids']
        counter = Counter(tokens)
        self.unigrams = list(counter.keys())
        self.sample_unigrams = []
        self.sample_unigram_ptr = 0
        if sample_distr == 'unigram':
            self.unigram2prob = {unigram: counter[unigram] / len(tokens) for unigram in self.unigrams}
        elif sample_distr == 'uniform':
            self.unigram2prob = {unigram: 1 / len(self.unigrams) for unigram in self.unigrams}

    def sample(self):
        if self.sample_unigram_ptr >= len(self.sample_unigrams):
            self.sample_unigrams = np.random.choice(
                self.unigrams, 10 ** 8, p=[self.unigram2prob[unigram] for unigram in self.unigrams])
            self.sample_unigram_ptr = 0
        unigram = self.sample_unigrams[self.sample_unigram_ptr]
        self.sample_unigram_ptr += 1
        return unigram


class RandomNumberGenerator():
    def __init__(self, p):
        # return 0 (1-p) or 1 (p)
        self.p = p
        self.numbers = []
        self.ptr = 0

    def sample(self):
        if self.ptr >= len(self.numbers):
            self.numbers = np.random.uniform(0, 1, 10 ** 8) <= self.p
            self.ptr = 0
        num = self.numbers[self.ptr]
        self.ptr += 1
        return num


def calculate_threshold(exidx2out, perturb_tokenidxs, perturb_fraction, f_log, skip_exidx_tokenidxs=None):
    if skip_exidx_tokenidxs is not None:
        skip_exidx_tokenidxs = set(skip_exidx_tokenidxs)
    exidx_perturbtokenidx2strength = {}
    for exidx in tqdm(exidx2out):
        token_strengths = exidx2out[exidx]
        for tokenidx in perturb_tokenidxs:
            assert not np.isnan(token_strengths[tokenidx])
            if (skip_exidx_tokenidxs is None) or ((exidx, tokenidx) not in skip_exidx_tokenidxs):
                exidx_perturbtokenidx2strength[(exidx, tokenidx)] = token_strengths[tokenidx]
    # check if the exidx_perturbtokenidx2strength has values of type int. if so, add noise (-0.1, 0.1) otherwise multiple examples correspond to the selected threshold.
    strengths = list(exidx_perturbtokenidx2strength.values())
    strength_is_int = np.all([type(strength) == int for strength in random.sample(strengths, 20)])
    f_log.write(f'scores are integers: {strength_is_int}\n')
    f_log.flush()
    if strength_is_int:
        print(strength_is_int, flush=True)
        exidx_perturbtokenidx2strength = {key: exidx_perturbtokenidx2strength[key] + random.uniform(-0.1, 0.1) for key in tqdm(exidx_perturbtokenidx2strength)}
    # calculate threshold
    strengths = [exidx_perturbtokenidx2strength[exidx_tokenidx] for exidx_tokenidx in exidx_perturbtokenidx2strength.keys()]
    num_perturbed_tokens_include_skipped = len(exidx2out) * len(perturb_tokenidxs)
    if skip_exidx_tokenidxs is None:
        assert num_perturbed_tokens_include_skipped == len(exidx_perturbtokenidx2strength)
    else:
        assert num_perturbed_tokens_include_skipped > len(exidx_perturbtokenidx2strength)
    f_log.write(f'calculating threshold, perturb {perturb_fraction}, total perturbed tokens {perturb_fraction} x {num_perturbed_tokens_include_skipped}, total tokens that allows perturbation {len(exidx_perturbtokenidx2strength)} tokens\n')
    f_log.flush()
    qs = QuickSelector()
    thres = qs.findKthLargest(strengths, int(perturb_fraction * num_perturbed_tokens_include_skipped))
    f_log.write(f'threshold calculated: {thres}\n')
    f_log.flush()
    if thres < 0:
        raise Exception('not enough tokens where perplexity decreases after SGD')
    return thres, exidx_perturbtokenidx2strength


def perturb_data(data, num_examples, exidx_perturbtokenidx2strength, thres, perturb_method, f_log, replace_token_distr=None):
    # NOTE: exidx_perturbtokenidx2strength only contains tokens that can be perturbed!! Those tokens that cannot be perturbed are not in the dict.
    assert perturb_method in ['replace', 'mask']
    if perturb_method == 'replace':
        assert replace_token_distr in ['unigram', 'uniform']
        unigram_sampler = UnigramSampler(data.select(range(50000)), sample_distr=replace_token_distr)
    perturbed_data = []
    num_perturbed_tokens = 0
    perturbed_exidx_tokenidxs = []
    for exidx in tqdm(range(num_examples)):
        input_ids = deepcopy(data[exidx]['input_ids'])
        if perturb_method == 'mask':
            masked_tokenidxs = []
        for tokenidx in range(len(input_ids)):
            if ((exidx, tokenidx) in exidx_perturbtokenidx2strength) and exidx_perturbtokenidx2strength[(exidx, tokenidx)] > thres:
                if perturb_method == 'replace':
                    input_ids[tokenidx] = unigram_sampler.sample()
                else:
                    masked_tokenidxs.append(tokenidx)
                perturbed_exidx_tokenidxs.append((exidx, tokenidx))
                num_perturbed_tokens += 1
        if perturb_method == 'replace':
            perturbed_data.append({'input_ids': input_ids})
        else:
            perturbed_data.append({'input_ids': input_ids, 'masked_tokenidxs': masked_tokenidxs})
    assert len(perturbed_data) == num_examples
    f_log.write(f'perturbing data: {num_examples}, {num_perturbed_tokens}\n')
    f_log.flush()
    return perturbed_data, perturbed_exidx_tokenidxs


def ablate_pretraining_data_structure(data, exidx2out, perturb_method, perturb_fraction, perturb_tokenidxs, outdir, replace_token_distr=None):
    assert not os.path.exists(outdir)
    os.makedirs(outdir, exist_ok=True)
    f_log = open(f'{outdir}/log.txt', 'w')
    f_log.write(f'{perturb_method}, {perturb_fraction}, {perturb_tokenidxs}, {replace_token_distr}\n')
    # check that exidx2out keys is 0-len(exidx2out)
    assert set(exidx2out.keys()) == set(range(len(exidx2out)))
    data = data.select(range(len(exidx2out)))
    thres, exidx_perturbtokenidx2strength = calculate_threshold(exidx2out, perturb_tokenidxs, perturb_fraction, f_log)
    perturbed_data, perturbed_exidx_tokenidxs = perturb_data(data, len(exidx2out), exidx_perturbtokenidx2strength, thres, perturb_method, f_log, replace_token_distr)
    num_perturbed_tokens = len(perturbed_exidx_tokenidxs)
    num_perturb_candidate_tokens = len(data) * len(perturb_tokenidxs)
    print(num_perturbed_tokens, num_perturb_candidate_tokens, perturb_fraction)
    assert np.abs(num_perturbed_tokens / num_perturb_candidate_tokens - perturb_fraction) < 0.0001
    print('saving data', flush=True)
    Dataset.from_list(perturbed_data).save_to_disk(outdir)
    pkl.dump(thres, open(f'{outdir}/perturb_threshold.pkl', 'wb'))
    pkl.dump(perturbed_exidx_tokenidxs, open(f'{outdir}/perturbed_exidx_tokenidxs.pkl', 'wb'))


def ablate_pretraining_data_structure_diff(data, exidx2out1, exidx2out2, perturb_method, perturb_fraction, perturb_tokenidxs, outdir, replace_token_distr=None):
    # structure1 \ structure2; then select top-n tokens based on structure1
    assert not os.path.exists(outdir)
    os.makedirs(outdir, exist_ok=True)
    f_log = open(f'{outdir}/log.txt', 'w')
    f_log.write(f'{perturb_method}, {perturb_fraction}, {perturb_tokenidxs}, {replace_token_distr}\n')
    num_examples = min(len(exidx2out1), len(exidx2out2))
    exidx2out1, exidx2out2 = {exidx: exidx2out1[exidx] for exidx in range(num_examples)}, {exidx: exidx2out2[exidx] for exidx in range(num_examples)}
    data = data.select(range(num_examples))
    # first filter out tokens from structure2
    f_log.write('calculating structure 2...\n')
    structure2_threshold, exidx_perturbtokenidx2strength2 = calculate_threshold(exidx2out2, perturb_tokenidxs, perturb_fraction, f_log)
    structure2_exidx_perturbtokenidxs = [key for key in exidx_perturbtokenidx2strength2 if exidx_perturbtokenidx2strength2[key] > structure2_threshold]
    # then select top-n tokens based on structure1
    f_log.write('calculating structure 1...\n')
    thres, exidx_perturbtokenidx2strength1 = calculate_threshold(exidx2out1, perturb_tokenidxs, perturb_fraction, f_log, structure2_exidx_perturbtokenidxs)
    perturbed_data, perturbed_exidx_tokenidxs = perturb_data(data, num_examples, exidx_perturbtokenidx2strength1, thres, perturb_method, f_log, replace_token_distr)
    num_perturbed_tokens = len(perturbed_exidx_tokenidxs)
    num_perturb_candidate_tokens = len(data) * len(perturb_tokenidxs)
    print(num_perturbed_tokens, num_perturb_candidate_tokens, perturb_fraction)
    assert np.abs(num_perturbed_tokens / num_perturb_candidate_tokens - perturb_fraction) < 0.0001
    print('saving data', flush=True)
    Dataset.from_list(perturbed_data).save_to_disk(outdir)
    pkl.dump(perturbed_exidx_tokenidxs, open(f'{outdir}/perturbed_exidx_tokenidxs.pkl', 'wb'))
    f_log.close()


def ablate_pretraining_data_random(data, perturb_method, exidxs, perturb_fraction, perturb_tokenidxs, outdir, replace_token_distr=None):
    assert not os.path.exists(outdir)
    assert perturb_method in ['replace', 'mask']
    if perturb_method == 'replace':
        assert replace_token_distr in ['unigram', 'uniform']
        unigram_sampler = UnigramSampler(data.select(range(50000)), sample_distr=replace_token_distr)
    random_perturb_generator = RandomNumberGenerator(perturb_fraction)
    perturbed_data = []
    num_perturbed_tokens = 0
    for exidx in tqdm(exidxs):
        input_ids = deepcopy(data[exidx]['input_ids'])
        if perturb_method == 'mask':
            masked_tokenidxs = []
        for tokenidx in perturb_tokenidxs:
            if random_perturb_generator.sample():
                if perturb_method == 'replace':
                    input_ids[tokenidx] = unigram_sampler.sample()
                else:
                    masked_tokenidxs.append(tokenidx)
                num_perturbed_tokens += 1
        if perturb_method == 'replace':
            perturbed_data.append({'input_ids': input_ids})
        else:
            perturbed_data.append({'input_ids': input_ids, 'masked_tokenidxs': masked_tokenidxs})
    assert len(perturbed_data) == len(exidxs)
    print(len(exidxs), perturb_fraction, num_perturbed_tokens)
    Dataset.from_list(perturbed_data).save_to_disk(outdir)


def is_sublist(a, b):
    a = list(a)
    b = list(b)
    # check if a is in b
    if (len(a) > len(b)):
        return False
    b_unigrams = set(b)
    if (a[0] not in b_unigrams) or (a[1] not in b_unigrams):
        return False
    for start_pos in range(len(b) - len(a) + 1):
        sublist = b[start_pos: start_pos + len(a)]
        if sublist == a:
            return True
    return False


def ablate_parallel_minus_copy(data, ps_exidx2out, perturb_method, perturb_fraction, perturb_tokenidxs, outdir, replace_token_distr):
    # for all (window i, token j) pairs, if (j-1, j) in window i, set the score of (window i, token j) to be 0.
    # aggregate the (window i, token j) pairs across all window i's, to get the score of each token j.
    # follow the standard procedure to do ablation.
    ps_exidx2out = {exidx: ps_exidx2out[exidx]['numepochs2out'][1]['orig_losses'] - ps_exidx2out[exidx]['numepochs2out'][1]['trained_losses'] for exidx in ps_exidx2out}
    exidxs = list(ps_exidx2out.keys())
    exidx2maxlossdecrease = {}
    for exidx in exidxs:
        out = ps_exidx2out[exidx]
        assert out.shape[1] == 1
        window_token_scores = out[:, 0, :]
        assert len(window_token_scores.shape) == 2
        ex = data[exidx]['input_ids']
        for window_idx in range(7):
            for token_idx in range(128, 1024):
                if (not np.isnan(window_token_scores[window_idx][token_idx])) and is_sublist((ex[token_idx-1], ex[token_idx]), ex[max(0, 128 * window_idx - 1): 128 * (window_idx+1)]):
                    window_token_scores[window_idx][token_idx] = -np.inf
        exidx2maxlossdecrease[exidx] = np.nanmax(window_token_scores, axis=0).tolist()
    exidx2out = exidx2maxlossdecrease
    # ablate
    assert not os.path.exists(outdir)
    os.makedirs(outdir, exist_ok=True)
    f_log = open(f'{outdir}/log.txt', 'w')
    f_log.write(f'{perturb_method}, {perturb_fraction}, {perturb_tokenidxs}, {replace_token_distr}\n')
    # check that exidx2out keys is 0-len(exidx2out)
    assert set(exidx2out.keys()) == set(range(len(exidx2out)))
    data = data.select(range(len(exidx2out)))
    thres, exidx_perturbtokenidx2strength = calculate_threshold(exidx2out, perturb_tokenidxs, perturb_fraction, f_log)
    perturbed_data, perturbed_exidx_tokenidxs = perturb_data(data, len(exidx2out), exidx_perturbtokenidx2strength, thres, perturb_method, f_log, replace_token_distr)
    num_perturbed_tokens = len(perturbed_exidx_tokenidxs)
    num_perturb_candidate_tokens = len(data) * len(perturb_tokenidxs)
    print(num_perturbed_tokens, num_perturb_candidate_tokens, perturb_fraction)
    assert np.abs(num_perturbed_tokens / num_perturb_candidate_tokens - perturb_fraction) < 0.0001
    print('saving data', flush=True)
    Dataset.from_list(perturbed_data).save_to_disk(outdir)
    pkl.dump(thres, open(f'{outdir}/perturb_threshold.pkl', 'wb'))
    pkl.dump(perturbed_exidx_tokenidxs, open(f'{outdir}/perturbed_exidx_tokenidxs.pkl', 'wb'))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--perturb_fraction", type=float, default=None, required=True)
    parser.add_argument("--data_dir", type=str, default=None, required=True)
    parser.add_argument("--out_dir", type=str, default=None, required=True)
    parser.add_argument("--setting", type=str, choices=['one_structure', 'two_structure_diff', 'one_structure_minus_repetition'])
    parser.add_argument("--structure1_detected_scores_fname", type=str, default=None, required=True)
    parser.add_argument("--structure2_detected_scores_fname", type=str, default=None, required=False)
    args = parser.parse_args()
    
    perturb_method = 'replace'
    replace_token_distr = 'uniform'
    perturb_tokenidxs = range(128,1024)    
    data = load_from_disk(args.data_dir)
    
    if args.setting == 'one_structure':
        detected_scores = pkl.load(open(args.structure1_detected_scores_fname))
        ablate_pretraining_data_structure(data, detected_scores, perturb_method, args.perturb_fraction, perturb_tokenidxs, args.out_dir, replace_token_distr)
    
    elif args.setting == 'two_structure_diff':
        struct1_detected_scores = pkl.load(open(args.structure1_detected_scores_fname))
        struct2_detected_scores = pkl.load(open(args.structure2_detected_scores_fname))
        ablate_pretraining_data_structure_diff(data, struct1_detected_scores, struct2_detected_scores, perturb_method, args.perturb_fraction, perturb_tokenidxs, args.out_dir, replace_token_distr)
    
    elif args.setting == 'one_structure_minus_repetition':
        detected_scores = pkl.load(open(args.structure1_detected_scores_fname))
        ablate_parallel_minus_copy(data, detected_scores, perturb_method, args.perturb_fraction, perturb_tokenidxs, args.out_dir, replace_token_distr)

    else:
        raise NotImplementedError