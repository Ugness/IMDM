checkpoint_path=${1:-"./outputs/owt/mdlm.ckpt"}

datasets=("ag_news"
          "scientific_papers_pubmed"
          "scientific_papers_arxiv"
          "lambada"
          "wikitext2"
          "wikitext103"
          "ptb"
          "lm1b-gpt2")
for data in "${datasets[@]}"; do
  echo "$data"
  generated_samples_path=./results/owt/ppl/MDLM-$data-gpt2.json
  echo "  Generated samples path: $generated_samples_path"
  python -u -m main \
    mode=ppl_eval \
    loader.eval_batch_size=16 \
    loader.eval_global_batch_size=128 \
    data="$data" \
    data.insert_valid_eos=False \
    model=small \
    model.length=1024 \
    model.softcap=-1 \
    algo=imdm \
    algo.noise_type=rand \
    eval.checkpoint_path=$checkpoint_path \
    eval.generated_samples_path=$generated_samples_path \
    sampling.predictor=ancestral_cache \
    sampling.temperature=1.0 \
    sampling.noise_removal=ancestral \
    +wandb.offline=true \
    eval.gen_ppl_eval_model_name_or_path=gpt2-large \
    sampling.num_sample_batches=0 \
    algo.as_mdlm=True \
    sampling.mdlm_posterior=True
done