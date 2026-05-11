checkpoint_path=${1:-"./outputs/owt/imsdtt_noisedim2048.ckpt"}
noise_dim=${2:-2048}
exp_name="OWT-IM-SDTT"

python -u -m main \
    loader.batch_size=16 \
    loader.global_batch_size=128 \
    loader.eval_batch_size=16 \
    data=openwebtext-split \
    wandb.name=$exp_name \
    model=small \
    model.length=1024 \
    model.softcap=-1 \
    algo=im-sdtt \
    algo.noise_type=rand \
    trainer.max_steps=70001 \
    trainer.log_every_n_steps=50 \
    trainer.limit_val_batches=2 \
    training.load_ema=True \
    callbacks.checkpoint_every_n_steps.every_n_train_steps=10000 \
    sampling.predictor=ancestral_cache \
    sampling.steps=\'1,2,4,8,32,128\' \
    eval.gen_ppl_eval_model_name_or_path=gpt2-large \
    lr_scheduler.num_warmup_steps=500 \
    optim.lr=6e-5 \
    +trainer.val_check_interval=1000 \
    training.finetune_path=$checkpoint_path \
    model.noise_dim=$noise_dim \