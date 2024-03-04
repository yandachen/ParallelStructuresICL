# Parallel Structures in Pre-training Data Yield In-Context Learning

This is the implementation of the paper [Parallel Structures in Pre-training Data Yield In-Context Learning](https://arxiv.org/abs/2402.12530). 
## Table of Contents
* [Overview](#overview)
* [Requirements](#requirements)
* [Code Structure](#code-structure)
* [How to Cite](#citation)


## Overview
Pre-trained language models (LMs) are capable of in-context learning (ICL): they can adapt to a task with only a few examples given in the prompt without any parameter update. However, it is unclear where this capability comes from as there is a stark distribution shift between pre-training text and ICL prompts. In this work, we study what patterns of the pre-training data contribute to ICL. 

We find that LMs' ICL ability depends on ***parallel structures*** in the pre-training data -- pairs of phrases following similar templates in the same context window. Specifically, we detect parallel structures by checking whether training on one phrase improves prediction of the other, and conduct ablation experiments to study their effect on ICL. We show that removing parallel structures in the pre-training data reduces LMs' ICL accuracy by **51%** (vs 2% from random ablation). This drop persists even when excluding common patterns such as n-gram repetitions and long-range dependency, showing the diversity and generality of parallel structures. A closer look at the detected parallel structures indicates that they cover diverse linguistic tasks and span long distances in the data.

You could find more details of this work in our [paper](https://arxiv.org/abs/2402.12530).

## Requirements

To run our code, please install all the dependency packages by using the following command:
```
pip install -r requirements.txt
```

## Code Structure

- Data Ablation

    - `detect_parallel_structure.py` detects parallel structures in natural text. You can run the script by ```python detect_parallel_structure.py --gpu GPUs --model_name MODEL_NAME --data_dir DATA_DIR --train_window_size TRAINW --eval_window_size EVALW --start_exidx START_EXAMPLE_IDX --end_exidx END_EXAMPLE_IDX --num_processes_per_gpu NUM_PROCESSES_PER_GPU --num_epochs NUM_EPOCHS --lr LR --out_dir OUT_DIR```.

    - `detect_dependency.py` detects dependency in natural text (baseline). You can run the script by ```python detect_dependency.py --model_name MODEL_NAME --data_dir DATA_DIR --eval_window_size EVALW --out_dir OUT_DIR```.

    - `ablate_structure.py` ablates the pre-training corpus based on the detected structures (calculated by `detect_parallel_structure.py` or `detect_dependency.py`). You can run the script by ```python ablate_structure.py --perturb_fraction PERTURB_FRACTION --data_dir DATA_DIR --out_dir OUT_DIR -setting {one_structure, two_structure_diff, one_structure_minus_repetition} --structure1_detected_scores_fname STRUCTURE1_FNAME [--structure2_detected_scores_fname STRUCTURE2_FNAME]```.

- Pre-training
    - `modelwrapper.py` wraps transformer models for training and inference.
    - `pretrain.py` pretrains LMs on text. You can run the script by 
    ```python pretrain.py --train_data_fname TRAIN_DATA_FNAME --val_data_fname VAL_DATA_NAME --model_name MODEL_NAME --n_gpus N_GPUS --num_train_steps NUM_TRAIN_STEPS --num_warmup_steps NUM_WARMUP_STEPS --per_device_batch_size PER_DEVICE_BATCH_SIZE --effective_bsz EFFECTIVE_BSZ --lr LR --bf16 BF16 --patience_num_train_steps PATIENCE_NUM_TRAIN_STEPS --patience_perplexity_decrease PATIENCE_PERPLEXITY_DECREASE --out_dir OUT_DIR --log_every_steps LOG_EVERY_STEPS```.

- In-Context Learning Evaluation
    - `evaluate_icl.py` evaluates ICL accuracy of a LM. The data used for ICL evaluation is `./icl_eval_data.json`.

## Questions?

If you have any questions related to the code or the paper, feel free to reach out to us at `yanda.chen@cs.columbia.edu`.

## Citation

```bibtex
@article{chen2024parallel,
  title={Parallel Structures in Pre-training Data Yield In-Context Learning},
  author={Chen, Yanda and Zhao, Chen and Yu, Zhou and McKeown, Kathleen and He, He},
  journal={arXiv preprint arXiv:2402.12530},
  year={2024}
}
```