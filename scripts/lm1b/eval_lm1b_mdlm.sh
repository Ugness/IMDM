checkpoint_path=$1
generated_samples_path=./results/lm1b/ppl/MDLM-LM1B-gpt2.json

python -u -m main \
    mode=ppl_eval \
    loader.batch_size=64 \
    loader.eval_batch_size=64 \
    data=lm1b-wrap \
    model=small \
    model.length=128 \
    model.softcap=50 \
    algo=imdm \
    algo.as_mdlm=True \
    eval.checkpoint_path=$checkpoint_path \
    eval.generated_samples_path=$generated_samples_path \
    sampling.num_sample_batches=0 \
    sampling.predictor=ancestral_cache \
    eval.gen_ppl_eval_model_name_or_path=gpt2-large \
    +wandb.offline=true