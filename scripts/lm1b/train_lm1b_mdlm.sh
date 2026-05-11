exp_name="LM1B-MDLM"

python -u -m main \
    loader.batch_size=128 \
    loader.eval_batch_size=128 \
    data=lm1b-wrap \
    wandb.name=$exp_name \
    model=small \
    model.length=128 \
    model.softcap=50 \
    algo=imdm \
    algo.noise_type=rand \
    trainer.max_steps=1000000 \
    trainer.log_every_n_steps=50 \
    trainer.limit_val_batches=2 \
    training.load_ema=True \
    callbacks.checkpoint_every_n_steps.every_n_train_steps=10000 \
    sampling.predictor=ancestral_cache \
    sampling.steps=\'1,2,4,8,128\' \
    eval.gen_ppl_eval_model_name_or_path=gpt2-large \
    algo.as_mdlm=True \
    sampling.mdlm_posterior=True \
