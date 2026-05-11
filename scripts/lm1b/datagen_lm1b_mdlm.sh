checkpoint_path=${1:-"./outputs/lm1b/mdlm.ckpt"}
reflow_save_path=${2:-"./reflow_data/lm1b/mdlm"}
steps=${3:-1024}
num_reflow_samples=${4:-100000}

python -u -m main \
    mode=generate_reflow_dataset \
    seed=42 \
    eval.checkpoint_path=$checkpoint_path \
    loader.batch_size=128 \
    loader.eval_batch_size=128 \
    data=lm1b-wrap \
    +wandb.offline=true \
    model=small \
    model.length=128 \
    model.softcap=50 \
    algo=imdm \
    algo.noise_type=rand \
    sampling.predictor=ancestral_cache \
    reflow.save_path=$reflow_save_path \
    sampling.steps=$steps \
    reflow.num_reflow_samples=$num_reflow_samples \
    eval.gen_ppl_eval_model_name_or_path=gpt2-large \
    algo.as_mdlm=True \
    sampling.mdlm_posterior=True \