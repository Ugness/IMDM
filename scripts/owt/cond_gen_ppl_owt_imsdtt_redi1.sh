steps=${1:-1024}
checkpoint_path=${2:-"./outputs/owt/imsdtt_noisedim2048_redi1.ckpt"}
output_path=${3:-"./results/owt"}
noise_dim=${4:-2048}

generated_samples_path=$output_path/cond_samples/$steps-IMSDTT_ReDi1-gpt2.json

python -u -m main \
        mode=cond_sample_eval_with_tc \
        seed=42 \
        loader.eval_batch_size=8 \
        data=openwebtext-split \
        model=small \
        model.length=1024 \
        model.softcap=-1 \
        algo=imdm \
        algo.noise_type=rand \
        eval.checkpoint_path=$checkpoint_path \
        sampling.num_sample_batches=32 \
        sampling.steps=$steps \
        sampling.predictor=ancestral_cache \
        sampling.duplicate=1 \
        +wandb.offline=true \
        eval.generated_samples_path=$generated_samples_path \
        sampling.temperature=1.0 \
        sampling.noise_removal=ancestral \
        eval.gen_ppl_eval_model_name_or_path=gpt2-large \
        cond_sampling.num_cont_per_prefix=5 \
        model.noise_dim=$noise_dim \