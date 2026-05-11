steps=${1:-1024}
checkpoint_path=${2:-"./outputs/lm1b/imsdtt_uniform_redi1.ckpt"}
output_path=${3:-"./results/lm1b"}
noise_scale=${4:-1.0}
noise_dim=${5:-768}

generated_samples_path=$output_path/samples/${steps}-IMSDTT_uniform_ReDi1-gpt2.json

python -u -m main \
        mode=sample_eval_with_tc \
        seed=42 \
        loader.eval_batch_size=64 \
        data=lm1b-wrap \
        model=small \
        model.length=128 \
        model.softcap=50 \
        algo=imdm \
        algo.noise_type=rand \
        eval.checkpoint_path=$checkpoint_path \
        sampling.num_sample_batches=16 \
        sampling.steps=$steps \
        sampling.predictor=ancestral_cache \
        sampling.duplicate=1 \
        +wandb.offline=true \
        eval.generated_samples_path=$generated_samples_path \
        sampling.temperature=1.0 \
        sampling.noise_removal=ancestral \
        eval.gen_ppl_eval_model_name_or_path=gpt2-large \
        model.noise_dim=$noise_dim \
        algo.noise_scale=$noise_scale \