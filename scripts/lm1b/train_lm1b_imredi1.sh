ckpt_path=${1:-"./outputs/lm1b/mdlm.ckpt"}
reflow_save_path=${2:-"./reflow_data/lm1b/mdlm"}
exp_name="LM1B-IM-REDI1"

python -u -m main \
    loader.batch_size=128 \
    loader.eval_batch_size=128 \
    data=reflow-dataset-general \
    data.tokenizer_name_or_path=bert-base-uncased \
    wandb.name=$exp_name \
    model=small \
    model.length=128 \
    model.softcap=50 \
    algo=im-redi \
    algo.noise_type=rand \
    algo.ignore_seed=True \
    trainer.max_steps=60000 \
    trainer.log_every_n_steps=50 \
    trainer.limit_val_batches=2 \
    training.load_ema=True \
    callbacks.checkpoint_every_n_steps.every_n_train_steps=10000 \
    sampling.predictor=ancestral_cache \
    sampling.steps=\'1,2,4,8,32,128\' \
    eval.gen_ppl_eval_model_name_or_path=gpt2-large \
    reflow.save_path=$reflow_save_path \
    training.finetune_path=$ckpt_path \
