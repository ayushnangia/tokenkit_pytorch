#!/bin/bash

# Example script for mined mapping-based distillation from Llama3 to Qwen2 tokenizer
# This demonstrates using the baseline_mined loss function which uses pre-computed MinED token mappings

# Before running this script, you'll need to:
# 1. Compute tokenizer statistics:
# ipython --pdb scripts/compute_tokenizer_info.py -- \
#     teacher_tokenizer_name=\'meta-llama/Llama-3.2-3B-Instruct:source=Llama3\' \
#     target_tokenizer_name=\'Qwen/Qwen2.5-1.5B:source=Qwen2:target=Llama3\' \
#     output=\'outputs/tokenizer_data/llama3_to_qwen2\'
#
# 2. Compute mined mappings:
# ipython --pdb scripts/compute_mined_mapping.py -- \
#     teacher_tokenizer_name=\'meta-llama/Llama-3.2-3B-Instruct:source=Llama3\' \
#     target_tokenizer_name=\'Qwen/Qwen2.5-1.5B:source=Qwen2:target=Llama3\' \
#     output=\'outputs/tokenizer_data/llama3_to_qwen2_mined\'

if [ "$1" == "debug" ]; then
    export CUDA_VISIBLE_DEVICES=0
    export WANDB_MODE=disabled
    DEBUG=true
    NUM_DEVICES=1
else
    DEBUG=false
    NUM_DEVICES=8
fi

echo "DEBUG: $DEBUG"
echo "Using $NUM_DEVICES devices"

NAME=llama3_to_qwen2_mined
ipython --pdb scripts/cross_tokenizer_distill.py -- \
    losses=[baseline_mined] \
    debug=$DEBUG \
    baseline.divergence=srkl \
    baseline.teacher_temperature=1.0 \
    max_teacher_length=512 \
    max_student_length=512 \
    n_data_parallel=$NUM_DEVICES \
    n_model_parallel=1 \
    steps=5000 \
    eval_interval=1000 \
    save_interval=1000 \
    optimizer.learning_rate=1e-5 \
    optimizer.weight_decay=0.0 \
    optimizer.max_grad_norm=null \
    optimizer.grad_acc_steps=4 \
    eval.tasks=[arc_easy,arc_challenge,piqa,hellaswag,boolq,arithmetic,mmlu] \
    eval.lengths=[128,256,512,1024,2048] \
    eval.tokens_per_batch=8192 \
    eval.add_bos=true \
    data.batch_size=64 \
    data.num_workers=16 \
    ppl_eval_data.batch_size=64 \
    +student.pretrained_model_name_or_path="benjamin/Llama-3.2-3B-Instruct-flax" \
    +student.tokenizer_name=\'meta-llama/Llama-3.2-3B-Instruct:source=Llama3\' \
    +target_tokenizer_name=\'Qwen/Qwen2.5-1.5B:source=Qwen2:target=Llama3\' \
    tokenizer_pair_data_path=\'outputs/tokenizer_data/llama3_to_qwen2\' \
    mined_mapping_path=\'outputs/tokenizer_data/llama3_to_qwen2_mined/mapping.pt\' \
    num_workers=16 \
    name=$NAME