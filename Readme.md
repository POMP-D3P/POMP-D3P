# Making Better Decision by  Directly Planning  in Continuous Control
This repository contains the code for *Making Better Decision by  Directly Planning  in Continuous Control* (POMP).

## Requirements and Installation
- PyTorch=1.12.1
- functorch=0.2.1
- gym

## Training
This is an example scripts to reproduce our experiment on `Ant`.
```shell
python main_maac2.py --exploration_init --cuda --save_result --save_model \
    --automatic_entropy_tuning True --see_freq 1000 \
    --env-name AntTruncatedObs-v2 --num_steps 150000 \
    --start_steps 5000 --save_model_interval 1000 \
    --model_type Naive --weight_grad 10 \
    --batch_size_pmp 256 --lr 3e-4 \
    --update_policy_times 10 --updates_per_step 10 \
    --rollout_max_length 1 --max_train_repeat_per_step 10 --min_pool_size 5000 \
    --near_n 5 --seed {seed} --H 4 \
    --save_prefix save0929 --policy_direct_bp \
    --ddp_max_delta 20 --ddp_clipk 0.025 --ddp_delta_decay_legacy
```

## Planning
Please use our script to load your pre-trained checkpoint for planning.
```shell
# ckt_dir is the directory you saved checkpoints
# step is the corresponding training step
python evaluate_ddp_offline.py {ckt_dir} \
    --iter-range start0end5 --step {step} --logpi-alpha 0.1
```