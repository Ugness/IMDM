checkpoint_path=${1:-"./outputs/owt_large/imsdtt_noisedim2048.ckpt"}
reflow_save_path=${2:-"./reflow_data/owt_large/imsdtt_noisedim2048"}
steps=${3:-1024}
num_reflow_samples=${4:-100000}
noise_dim=${5:-2048}

python -u -m main \
    mode=generate_reflow_dataset \
    seed=42 \
    eval.checkpoint_path=$checkpoint_path \
    loader.batch_size=32 \
    loader.eval_batch_size=32 \
    data=openwebtext-split \
    +wandb.offline=true \
    model=large \
    model.length=1024 \
    model.softcap=-1 \
    algo=imdm \
    algo.noise_type=rand \
    sampling.predictor=ancestral_cache \
    reflow.save_path=$reflow_save_path \
    sampling.steps=$steps \
    reflow.num_reflow_samples=$num_reflow_samples \
    model.noise_dim=$noise_dim \
    eval.gen_ppl_eval_model_name_or_path=gpt2-large \
