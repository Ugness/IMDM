<div align="center">

# Infinite Mask Diffusion for Few-Step Distillation

**[Jaehoon Yoo*](https://sites.google.com/view/jaehoon-yoo/홈)**, **[Wonjung Kim*](https://github.com/wjhjkim)**, **[Chanhyuk Lee](https://david3684.github.io)**, **[Seunghoon Hong](https://maga33.github.io/)**

KAIST

**[[Project Page]](https://Ugness.github.io/official_imdm)** | **[[Paper]](https://arxiv.org/abs/2600.00000)** | **[[Checkpoint]](https://huggingface.co/Ugness/IMDM)**

<!-- <a href="https://arxiv.org/abs/2600.00000"><img src="https://img.shields.io/badge/arXiv-IMDM-red" alt="arXiv"></a>
<a href="https://Ugness.github.io/official_imdm"><img src="https://img.shields.io/badge/Project_Page-IMDM-blue" alt="Project Page"></a>
<a href="https://huggingface.co/Ugness/IMDM"><img src="https://img.shields.io/badge/🤗-Huggingface-blue" alt="Huggingface"></a> -->

</div>

## TL;DR

We propose **Infinite Mask Diffusion Model**, which leverages the simple design and effective conditional generation of Masked Diffusion Models while overcoming their theoretical lower bound of factorization error.

## Overview

Masked Diffusion Models (MDMs) have emerged as a promising alternative to autoregressive models in language modeling, offering the advantages of parallel decoding and bidirectional context processing within a simple yet effective framework.
Specifically, their explicit distinction between masked tokens and data underlies their simple framework and effective conditional generation.
However, MDMs typically require many sampling iterations due to factorization errors stemming from simultaneous token updates.
We observe that a theoretical lower bound of the factorization error exists, which standard MDMs cannot reduce due to their use of a deterministic single-state mask.
In this paper, we propose the **Infinite Mask Diffusion Model (IMDM)**, which introduces a stochastic infinite-state mask to mitigate the theoretical bound while directly inheriting the benefits of MDMs, including the compatibility with pre-trained weights.
We empirically demonstrate that MDM fails to perform few-step generation even in a simple synthetic task due to the factorization error bound, whereas IMDM can find an efficient solution for the same task.
Finally, when equipped with appropriate distillation methods, IMDM surpasses existing few-step distillation methods at small step counts on LM1B and OpenWebText.

## Project Structure
      ├── config/                                <- Config files for datasets/denoising networks/noise schedules/LR schedules.
      |      └── config.yaml                     <- Main config file
      |
      ├── integral/
      |    
      ├── models/                                <- Denoising network architectures. Supports [DiT](https://arxiv.org/abs/2212.09748) and AR transformer.
      |      ├── dit.py                          <- DiT structure
      |      ├── ema.py                          <- EMA model
      |      └── unit_test_attention.py          <- Attention module
      |
      ├── scripts/                               <- Shell scripts for training/evaluation.
      |      ├── lm1b                            <- Shell scripts for LM1B dataset
      |      ├── owt                             <- Shell scripts for OpenWebText dataset
      |      └── owt_large                       <- Shell scripts for OpenWebText dataset with 860M large models
      |
      ├── algo.py                                <- Main model structures: Algorithms such as DUO, MDLM, AR, SEDD, D3PM, ReDi, IMDM, IM-SDTT, IM-ReDi.
      ├── dataloader.py                          <- Dataloader and tokenizer module
      ├── eval_mauve.py                          <- Eval the MAUVE score from cond. generation samples
      ├── LICENSE                                <- Apache License 2.0
      ├── main.py                                <- Main
      ├── metrics.py                             <- Metrics module
      ├── README.md                              
      ├── requirements.txt                       <- Help to install env 
      ├── trainer_base.py                        <- Boiler plate trainer using pytorch lightning.
      └── utils.py                               <- LR scheduler, logging, `fsspec` handling.

## Usage
To get started, follow these steps:

1. Install requirement
    ```bash
    pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0
    pip install -r requirements.txt
    pip install flash-attn==2.7.4.post1 --no-build-isolation
    ```

2. Download Pretrained models
    ```bash
    # 1. OWT Finetuned models(IM-SDTT-ReDi1, IM-SDTT-ReDi1 large)
    # Download from Hugginface(https://huggingface.co/Ugness/IMDM)

    # 2. Pretrained models from MDLM paper(OWT)
    # Download official MDLM checkpoint from Google Drive folder(https://drive.google.com/drive/folders/1JpqFM8XRvifwIkjWPfMyuDvu41r1yk0t?usp=share_link).

    # put checkpoints into ./outputs/lm1b, ./outputs/owt, ./outputs/owt_large.
    ```

3. Download LM1B, OpenWebText dataset
    ```bash
    # The training code automatically downloads the LM1B, OWT dataset onto your local(./cache/).

    # Or, set LM1B and OpenWebText dataset to the cache dir(./cache/).
    ```

4. Use IMDM
    ```bash
    # LM1B
    ## Train
    bash scripts/lm1b/train_lm1b_mdlm.sh
    bash scripts/lm1b/train_lm1b_imdm.sh
    bash scripts/lm1b/train_lm1b_imsdtt_uniform.sh              (required MDLM ckpt)
    bash scripts/lm1b/train_lm1b_imredi1.sh                     (required MDLM ckpt, Reflow dataset)
    bash scripts/lm1b/train_lm1b_imsdtt_uniform_redi1.sh        (required IM-SDTT ckpt, Reflow dataset)
    
    ### Create Rectified Coupling (perturbed ReDi)
    bash scripts/lm1b/datagen_lm1b_mdlm.sh                      (required MDLM ckpt)
    bash scripts/lm1b/datagen_lm1b_imsdtt_uniform.sh            (required IM-SDTT ckpt)
    
    ## Eval
    ### LM1B PPL eval
    bash scripts/lm1b/eval_lm1b_mdlm.sh
    bash scripts/lm1b/eval_lm1b_imdm.sh
    
    ### Uncond. generation eval
    bash scripts/lm1b/gen_ppl_lm1b_mdlm.sh
    bash scripts/lm1b/gen_ppl_lm1b_imsdtt_uniform.sh
    bash scripts/lm1b/gen_ppl_lm1b_imredi1.sh
    bash scripts/lm1b/gen_ppl_lm1b_imsdtt_uniform_redi1.sh


    # OpenWebText
    ## Train
    bash scripts/owt/train_owt_imsdtt.sh                        (required MDLM ckpt)
    bash scripts/owt/train_owt_imsdtt_redi1.sh                  (required IM-SDTT ckpt, Reflow dataset)
    
    ### Create Rectified Coupling (perturbed ReDi)
    bash scripts/owt/datagen_owt_imsdtt.sh                      (required IM-SDTT ckpt)
    
    ## Eval
    ### Zero-Shot PPL eval
    bash scripts/owt/zero_shot_mdlm.sh
    bash scripts/owt/zero_shot_imdm.sh
    
    ### Uncond. generation eval
    bash scripts/owt/gen_ppl_owt_mdlm.sh
    bash scripts/owt/gen_ppl_owt_imsdtt_redi1.sh
    
    ### Cond. generation eval
    bash scripts/owt/cond_gen_ppl_owt_mdlm.sh
    bash scripts/owt/cond_gen_ppl_owt_imsdtt_redi1.sh
    
    ### Eval MAUVE
    python eval_mauve.py --generation_path /path/to/your/cond/samples/json/file


    # OpenWebText(large model)
    ## Train
    bash scripts/owt_large/train_owt_imsdtt.sh                  (required MDLM ckpt)
    bash scripts/owt_large/train_owt_imsdtt_redi1.sh            (required IM-SDTT ckpt, Reflow dataset)

    ### Create Rectified Coupling (perturbed ReDi)
    bash scripts/owt_large/datagen_owt_imsdtt.sh                (required IM-SDTT ckpt)
    
    ## Eval
    ### Uncond. generation eval
    bash scripts/owt_large/gen_ppl_owt_imsdtt_redi1.sh
    
    ### Cond. generation eval
    bash scripts/owt_large/cond_gen_ppl_owt_imsdtt_redi1.sh
    
    ### Eval MAUVE
    python eval_mauve.py --generation_path /path/to/your/cond/samples/json/file
    ```

## Acknowledgments
This repository is built upon the codebases of **[Duo](https://github.com/s-sahoo/duo)** and **[ReDi](https://github.com/Ugness/ReDi)**.

## BibTeX
```bibtex
@inproceedings{yoo2026imdm,
      title={Infinite Mask Diffusion for Few-Step Distillation}, 
      author={Yoo, Jaehoon and Kim, Wonjung and Lee, Chanhyuk and Hong, Seunghoon},
      year={2026},
      booktitle={ICML}
}
```
