ckpt_path=${1:-"./outputs/owt_large/imsdtt_noisedim2048.ckpt"}
reflow_save_path=${2:-"./reflow_data/owt_large/imsdtt_noisedim2048"}
noise_dim=${3:-2048}
exp_name="OWT-LARGE-IM-SDTT-ReDi1"

python -u -m main \
    loader.batch_size=16 \
    loader.eval_batch_size=16 \
    data=reflow-dataset-general \
    data.tokenizer_name_or_path=gpt2 \
    wandb.name=$exp_name \
    model=large \
    model.length=1024 \
    model.softcap=-1 \
    algo=im-redi \
    algo.noise_type=rand \
    trainer.max_steps=30001 \
    trainer.log_every_n_steps=50 \
    trainer.limit_val_batches=2 \
    training.load_ema=True \
    callbacks.checkpoint_every_n_steps.every_n_train_steps=10000 \
    sampling.predictor=ancestral_cache \
    sampling.steps=\'1,2,4,8,32,128\' \
    eval.gen_ppl_eval_model_name_or_path=gpt2-large \
    reflow.save_path=$reflow_save_path \
    training.finetune_path=$ckpt_path \
    model.noise_dim=$noise_dim \