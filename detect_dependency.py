from modelwrapper import ModelWrapper
from transformers import AutoTokenizer
from datasets import load_from_disk, Dataset
import numpy as np
import pickle as pkl
import os


def detect_structure(mw, data_dir, short_context_window_size, out_dir, batch_size=1000):
    data = load_from_disk(data_dir)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    start_exidxs = range(0, len(data), batch_size)
    for start_exidx in start_exidxs:
        batch_exidxs = list(range(start_exidx, min(start_exidx + batch_size, len(data))))
        batch_outf = f'{out_dir}/{batch_exidxs[0]}-{batch_exidxs[-1]}.pkl'
        if os.path.exists(batch_outf):
            continue
        batch_data = data.select(batch_exidxs)
        entire_context_losses = mw.evaluate_perplexity(
            eval_data=batch_data, per_device_batch_size=32, use_accelerator=True, show_progress_bar=True)['token_losses']
        assert short_context_window_size % 2 == 0
        short_context_losses = mw.evaluate_perplexity_limit_context_size(eval_data=batch_data, window_size=short_context_window_size, stride=short_context_window_size // 2,
                                                                        per_device_batch_size=1024, use_accelerator=True, show_progress_bar=True)['token_losses']
        loss_diff = []
        assert len(entire_context_losses) == len(short_context_losses) == len(batch_data) == len(batch_exidxs)
        for ex_entire_context_loss, ex_short_context_loss in zip(entire_context_losses, short_context_losses):
            loss_diff.append((np.array(ex_short_context_loss) -
                            np.array(ex_entire_context_loss)).tolist())
        assert len(batch_exidxs) == len(loss_diff)
        exidx2diff = {}
        for exidx, diff in zip(batch_exidxs, loss_diff):
            exidx2diff[exidx] = diff
        if mw.accelerator.is_main_process:
            pkl.dump(exidx2diff, open(batch_outf, 'wb'))
    all_exidx2diff = {}
    for start_exidx in range(0, len(data), batch_size):
        batch_exidxs = range(start_exidx, min(start_exidx + batch_size, len(data)))
        batch_outf = f'{out_dir}/{batch_exidxs[0]}-{batch_exidxs[-1]}.pkl'
        all_exidx2diff.update(pkl.load(open(batch_outf, 'rb')))
    assert len(all_exidx2diff) == len(data)
    pkl.dump(all_exidx2diff, open(f'{out_dir}/exidx2out.pkl', 'wb'))



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--eval_window_size", type=int, default=None)
    parser.add_argument("--out_dir", type=str, default=None)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    mw = ModelWrapper(model_type='clm', model_name=args.model_name, tokenizer=tokenizer, load_pretrained=True,
                      load_pretrained_hf_name=args.model_name, bf16=False)
    detect_structure(mw, args.data_dir, args.eval_window_size,
                     out_dir=args.out_dir)
