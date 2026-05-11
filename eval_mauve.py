import mauve
import json
import math
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
LOG2 = math.log(2)
import dataloader
from hydra import initialize, compose
from datetime import datetime
import argparse

def main(seed, generation_path, sample_num, cache_dir, seq_len=100):
    num_sample_batches = 32
    batch_size = sample_num // num_sample_batches
    assert sample_num % num_sample_batches == 0

    samples = []
    with open(generation_path, "r") as f:
        file = json.load(f)
        for line in file["generated_seqs"]:
            samples.append(line)
    
    print("Seed: ", seed)
    print("Start time: ", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("Sample path: ", generation_path)

    with initialize(version_base=None, config_path="configs"):
        config = compose(config_name="config", overrides=[
                "loader.eval_batch_size="+str(batch_size),
                "data=openwebtext-split",
                "data.cache_dir="+cache_dir,
                "model=small",
                "sampling.num_sample_batches="+str(num_sample_batches),
                "sampling.predictor=ancestral_cache",
                "loader.num_workers=8",
                "seed="+str(seed),
                "eval.gen_ppl_eval_model_name_or_path=gpt2-large",
                "+wandb.offline=true"
                ])
        # compute MAUVE score
        tokenizer = dataloader.get_tokenizer(config)
        human_references = []
        np.random.seed(config.seed)
        _, valid_loader = dataloader.get_cond_dataloaders(
            config, tokenizer, min_seq_len=1024, skip_train=True, force=True)

        dataset = valid_loader.dataset['input_ids']
        shuffled_indices = np.random.permutation(len(dataset))[:sample_num]
        
        for i in range(num_sample_batches):
            batch = dataset[shuffled_indices[i * batch_size: (i + 1) * batch_size], :seq_len]
            human_references.extend(tokenizer.batch_decode(batch))
        
        print(f"Number of samples: {len(samples)}, Number of data: {len(human_references)}")
        print("samples[0:3]:", samples[0:3])
        print("--------------")
        print("data[0:3]:", human_references[0:3])
        print("Mauve test start time: ", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        start_time = datetime.now()
        assert len(human_references) * 5 == len(samples)
        mauve_score = []
        for i in range(5):
            results = mauve.compute_mauve(p_text=human_references, q_text=samples, seed=config.seed+i, max_text_length=seq_len, verbose=False)
            mauve_score.append(results.mauve)

        print(f"MAUVE: {np.mean(mauve_score):.4f}")
        print("End time: ", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        print("During time: ", datetime.now() - start_time)

        save_path = generation_path.replace("-gpt2.json", f"-mauve.json") if generation_path.endswith("-gpt2.json") else generation_path.replace("-llama3_1.json", f"-mauve.json")
        with open(save_path, "w") as f:
            json.dump({"mauve": np.mean(mauve_score)}, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--generation_path', type=str, default=None, help='Path to the generation json file.')
    parser.add_argument('--sample_num', type=int, default=256, help='Number of samples.')
    parser.add_argument('--cache_dir', type=str, default=" ./cache/openwebtext", help='Cache dir for transformers.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed used for generation.')
    parser.add_argument('--seq_len', type=int, default=100, help='Sequence length used for mauve computation.')

    args = parser.parse_args()

    main(seed=args.seed, generation_path=args.generation_path, sample_num=args.sample_num, cache_dir=args.cache_dir, seq_len=args.seq_len)