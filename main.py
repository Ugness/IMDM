import json
import os
from einops import rearrange
import fsspec
import hydra
import lightning as L
import omegaconf
import rich.syntax
import rich.tree
import torch
import functools
torch.load = functools.partial(torch.load, weights_only=False)

import algo
import dataloader
import utils

import numpy as np
from datetime import datetime

omegaconf.OmegaConf.register_new_resolver(
    'cwd', os.getcwd)
omegaconf.OmegaConf.register_new_resolver(
    'device_count', torch.cuda.device_count)
omegaconf.OmegaConf.register_new_resolver(
    'eval', eval)
omegaconf.OmegaConf.register_new_resolver(
    'div_up', lambda x, y: (x + y - 1) // y)


def _load_from_checkpoint(diffusion_model, config, tokenizer):
    if 'hf' in config.algo.backbone:
        return diffusion_model(
            config, tokenizer=tokenizer).to('cuda')
    
    return diffusion_model.load_from_checkpoint(
        config.eval.checkpoint_path,
        tokenizer=tokenizer,
        config=config, weights_only=False, strict=False)


@L.pytorch.utilities.rank_zero_only
def _print_config(
    config: omegaconf.DictConfig,
    resolve: bool = True,
    save_cfg: bool = True) -> None:
    """Prints content of DictConfig using Rich library and its tree structure.
    
    Args:
        config (DictConfig): Configuration composed by Hydra.
        resolve (bool): Whether to resolve reference fields of DictConfig.
        save_cfg (bool): Whether to save the configuration tree to a file.
    """

    style = 'dim'
    tree = rich.tree.Tree('CONFIG', style=style, guide_style=style)

    fields = config.keys()
    for field in fields:
        branch = tree.add(field, style=style, guide_style=style)

        config_section = config.get(field)
        branch_content = str(config_section)
        if isinstance(config_section, omegaconf.DictConfig):
            branch_content = omegaconf.OmegaConf.to_yaml(
                config_section, resolve=resolve)

        branch.add(rich.syntax.Syntax(branch_content, 'yaml'))
    rich.print(tree)
    if save_cfg:
        with fsspec.open(
            '{}/config_tree.txt'.format(
                config.checkpointing.save_dir), 'w') as fp:
            rich.print(tree, file=fp)


@L.pytorch.utilities.rank_zero_only
def _print_batch(train_ds, valid_ds, tokenizer, k=64):
    for dl_type, dl in [
        ('train', train_ds), ('valid', valid_ds)]:
        print(f'Printing {dl_type} dataloader batch.')
        batch = next(iter(dl))
        print('Batch input_ids.shape', batch['input_ids'].shape)
        first = batch['input_ids'][0, :k]
        last = batch['input_ids'][0, -k:]
        print(f'First {k} tokens:', tokenizer.decode(first))
        print('ids:', first)
        print(f'Last {k} tokens:', tokenizer.decode(last))
        print('ids:', last)

def _generate_samples(diffusion_model, config, logger,
                                            tokenizer):
    logger.info('Starting Sample Eval.')
    model = _load_from_checkpoint(
        diffusion_model=diffusion_model,
        config=config,
        tokenizer=tokenizer)
    model.metrics.gen_ppl.reset()
    model.metrics.sample_entropy.reset()
    if config.eval.disable_ema:
        logger.info('Disabling EMA.')
        model.ema = None
    stride_length = config.sampling.stride_length
    num_strides = config.sampling.num_strides
    all_samples = []

    print("generation start: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    start_time = datetime.now()
    
    for _ in range(config.sampling.num_sample_batches):
        if config.sampling.semi_ar:
            _, intermediate_samples, _ = model.restore_model_and_semi_ar_sample(
                stride_length=stride_length,
                num_strides=num_strides,
                dt=1 / config.sampling.steps)
            text_samples = intermediate_samples[-1]
        else:
            samples = model.restore_model_and_sample(
                num_steps=config.sampling.steps)
            model.metrics.record_entropy(samples)
            text_samples = model.tokenizer.batch_decode(samples)
            model.metrics.record_generative_perplexity(
                text_samples, config.model.length, model.device)
            all_samples.extend(list(text_samples))
    
    print("generation end: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    end_time = datetime.now()
    print("Total generation time: ", str(end_time - start_time))

    generative_ppl = 0.
    entropy = 0.
    if not config.sampling.semi_ar:
        generative_ppl = model.metrics.gen_ppl.compute().item()
        entropy = model.metrics.sample_entropy.compute().item()
        print('Generative perplexity:', generative_ppl)
        print('Sample entropy:', entropy)
    samples_path = config.eval.generated_samples_path
    with fsspec.open(samples_path, 'w') as f:
        json.dump({'generative_ppl': generative_ppl,
                             'entropy': entropy,
                             'generated_seqs': all_samples}, f, indent=4)
    print('Samples saved at:', samples_path)

def _generate_samples_with_tc(diffusion_model, config, logger,
                                            tokenizer):
    logger.info('Starting Sample Eval.')
    model = _load_from_checkpoint(
        diffusion_model=diffusion_model,
        config=config,
        tokenizer=tokenizer)
    model.metrics.gen_ppl.reset()
    model.metrics.sample_entropy.reset()
    if config.eval.disable_ema:
        logger.info('Disabling EMA.')
        model.ema = None
    stride_length = config.sampling.stride_length
    num_strides = config.sampling.num_strides
    all_samples = []

    print("generation start: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    start_time = datetime.now()
    
    for i in range(config.sampling.num_sample_batches):
        if config.sampling.semi_ar:
            _, intermediate_samples, _ = model.restore_model_and_semi_ar_sample(
                stride_length=stride_length,
                num_strides=num_strides,
                dt=1 / config.sampling.steps)
            text_samples = intermediate_samples[-1]
        else:
            assert config.loader.eval_batch_size % config.sampling.duplicate == 0
            different_in_batch = config.loader.eval_batch_size // config.sampling.duplicate
            samples = model.restore_model_and_sample(
                num_steps=config.sampling.steps, duplicate=config.sampling.duplicate, num_duplicate_batch=config.sampling.num_duplicate_batch if hasattr(config.sampling, 'num_duplicate_batch') else 1)
            model.metrics.record_entropy(samples)
            text_samples = model.tokenizer.batch_decode(samples)
            model.metrics.record_generative_perplexity(text_samples, config.model.length, model.device)
            model.metrics.record_tc([i*different_in_batch + j for _ in range(config.sampling.duplicate) for j in range(different_in_batch)], samples)
            all_samples.extend(list(text_samples))
    
    print("generation end: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    end_time = datetime.now()
    print("Total generation time: ", str(end_time - start_time))

    generative_ppl = 0.
    entropy = 0.
    if not config.sampling.semi_ar:
        generative_ppl = model.metrics.gen_ppl.compute().item()
        entropy = model.metrics.sample_entropy.compute().item()
        if config.sampling.duplicate > 1:
            avg_tc, _, _ = model.metrics.tc.compute()
        else:
            avg_tc = 0.
        print('Generative perplexity:', generative_ppl)
        print('Sample entropy:', entropy)
        print('Total average correlation:', avg_tc)
    samples_path = config.eval.generated_samples_path
    with fsspec.open(samples_path, 'w') as f:
        json.dump({
                             'total_generation_times': (end_time - start_time).total_seconds(),
                             'generative_ppl': generative_ppl,
                             'entropy': entropy,
                             'tc': avg_tc,
                             'generated_seqs': all_samples}, f, indent=4)
    print('Samples saved at:', samples_path)

def _generate_cond_samples_with_tc(diffusion_model, config, logger,
                                            tokenizer):
    logger.info('Starting Sample Eval.')
    model = _load_from_checkpoint(
        diffusion_model=diffusion_model,
        config=config,
        tokenizer=tokenizer)
    model.metrics.gen_ppl.reset()
    model.metrics.sample_entropy.reset()
    if config.eval.disable_ema:
        logger.info('Disabling EMA.')
        model.ema = None
    all_samples = []
    tensor_samples = []

    print("Loading conditional dataloader...")

    _, valid_bs = dataloader.get_cond_dataloaders(
        config, tokenizer, min_seq_len=config.cond_sampling.min_seq_len, skip_train=True, force=True)

    dataset = valid_bs.dataset['input_ids']
    subset_length = config.sampling.num_sample_batches * config.loader.eval_batch_size
    shuffled_indices = np.random.permutation(len(dataset))[:subset_length]
    shuffled_dataset = dataset[shuffled_indices]
    
    print("generation start: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    start_time = datetime.now()
    
    model._eval_mode()
    for i in range(config.sampling.num_sample_batches):
        if config.sampling.semi_ar:
            raise NotImplementedError("Semi-AR conditional sampling is not implemented. Please use standard sampling.")
        else:
            start_idx = i*config.loader.eval_batch_size
            end_idx = (i+1)*config.loader.eval_batch_size
            num_samples = shuffled_dataset[start_idx:end_idx].shape[0] * config.cond_sampling.num_cont_per_prefix
            samples = model.generate_cond_samples(
                num_samples=num_samples,
                data=shuffled_dataset[start_idx:end_idx].cuda().repeat(config.cond_sampling.num_cont_per_prefix, 1),
                prefix_len=config.cond_sampling.prefix_len,
                num_steps=config.sampling.steps,
                seed=torch.ones((num_samples, ), device=model.device).long() * config.cond_sampling.fix_seed if config.cond_sampling.fix_seed != -1 else None,
                )
            samples = samples[:, :config.cond_sampling.seq_len]
            rearranged_samples = samples.clone()
            if samples.ndim == 3:
                rearranged_samples = rearrange(rearranged_samples, "dev bs l -> (dev bs) l")
            tensor_samples.extend(list(torch.chunk(rearranged_samples.cpu(), chunks=config.cond_sampling.num_cont_per_prefix, dim=0)))
            model.metrics.record_entropy(samples)
            text_samples = model.tokenizer.batch_decode(samples)
            model.metrics.record_generative_perplexity(text_samples, config.model.length, model.device)
            all_samples.extend(list(text_samples))
    
    print("generation end: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    end_time = datetime.now()
    print("Total generation time: ", str(end_time - start_time))

    generative_ppl = 0.
    entropy = 0.
    if not config.sampling.semi_ar:
        generative_ppl = model.metrics.gen_ppl.compute().item()
        entropy = model.metrics.sample_entropy.compute().item()
        print('Generative perplexity:', generative_ppl)
        print('Sample entropy:', entropy)
    samples_path = config.eval.generated_samples_path
    with fsspec.open(samples_path, 'w') as f:
        json.dump({
                    'total_generation_times': (end_time - start_time).total_seconds(),
                    'generative_ppl': generative_ppl,
                    'entropy': entropy,
                    'seq_len': config.cond_sampling.seq_len,
                    'prefix_len': config.cond_sampling.prefix_len,
                    'generated_seqs': all_samples}, f, indent=4
                  )
    print('Samples saved at:', samples_path)

@torch.inference_mode()
def generate_reflow_dataset(diffusion_model, config, logger, tokenizer):
    logger.info('Starting Reflow Dataset Generation.')
    model = _load_from_checkpoint(
        diffusion_model=diffusion_model,
        config=config,
        tokenizer=tokenizer)
    model.metrics.gen_ppl.reset()
    model.metrics.sample_entropy.reset()
    if config.eval.disable_ema:
        logger.info('Disabling EMA.')
        model.ema = None
    
    train_ds, _ = dataloader.get_dataloaders(
        config, tokenizer, skip_valid=True, force=True)

    trainer = hydra.utils.instantiate(
        config.trainer,
        default_root_dir=config.checkpointing.save_dir,
        callbacks=None,
        strategy=hydra.utils.instantiate(config.strategy),
        logger=None)
    
    num_reflow_batches = config.reflow.num_reflow_samples // config.trainer.devices // config.loader.eval_batch_size
    modular = config.reflow.num_reflow_samples % (config.trainer.devices * config.loader.eval_batch_size)
    if modular != 0:
        num_reflow_batches += 1
        num_reflow_samples = num_reflow_batches * config.trainer.devices * config.loader.eval_batch_size
        logger.info(f'Adjusting num_reflow_samples to {num_reflow_samples} to fit batch size.')
    logger.info(f'Number of reflow batches: {num_reflow_samples}')
    subset_indices = np.random.choice(len(train_ds.dataset), size=num_reflow_samples, replace=False)
    logger.info(f'Subset indices length: {len(subset_indices)}')
    logger.info(f'train_ds num_workers: {train_ds.num_workers}')
    subset = torch.utils.data.Subset(train_ds.dataset, subset_indices)
    subset_loader = torch.utils.data.DataLoader(subset,
                                                batch_size=config.loader.eval_batch_size,
                                                shuffle=False,
                                                num_workers=train_ds.num_workers,
                                                collate_fn=train_ds.collate_fn,
                                                drop_last=True,
                                                pin_memory=True,
                                                )

    print("generation start: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    trainer.test(model, subset_loader)
 
def _eval_ppl(diffusion_model, config, logger, tokenizer):
    logger.info('Starting Perplexity Eval.')

    model = _load_from_checkpoint(
        diffusion_model=diffusion_model,
        config=config,
        tokenizer=tokenizer)
    if config.eval.disable_ema:
        logger.info('Disabling EMA.')
        model.ema = None

    wandb_logger = None
    if config.get('wandb', None) is not None:
        wandb_logger = L.pytorch.loggers.WandbLogger(
            config=omegaconf.OmegaConf.to_object(config),
            ** config.wandb)
    callbacks = []
    if 'callbacks' in config:
        for _, callback in config.callbacks.items():
            callbacks.append(hydra.utils.instantiate(callback))
    trainer = hydra.utils.instantiate(
        config.trainer,
        default_root_dir=config.checkpointing.save_dir,
        callbacks=callbacks,
        strategy=hydra.utils.instantiate(config.strategy),
        logger=wandb_logger)
    config.loader.drop_last = False
    _, valid_ds = dataloader.get_dataloaders(
        config, tokenizer, skip_train=True, valid_seed=config.seed)
    print("Validation dataloader loaded. Number of batches: ", len(valid_ds))
    dataset_length = []
    for i in range(len(valid_ds)):
        dataset_length.append(len(valid_ds.dataset[i]['input_ids']))
    print(f"Validation dataset length stats - min: {min(dataset_length)}, max: {max(dataset_length)}, mean: {sum(dataset_length)/len(dataset_length)}")
    print("Evaluation start: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    trainer.validate(model, valid_ds)
    print("Result of validate:", model.metrics.valid_nlls.ppl.compute().item())
    samples_path = config.eval.generated_samples_path
    with fsspec.open(samples_path, 'w') as f:
        json.dump({
                    'perplexity': model.metrics.valid_nlls.ppl.compute().item(),
                    }, f, indent=4)
    print("Evaluation end: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

def _train(diffusion_model, config, logger, tokenizer):
    logger.info('Starting Training.')
    wandb_logger = None
    if getattr(config.wandb, 'resume', None) == 'must':
        config.wandb.id = str(config.wandb.id)
        pass
    else:
        config.wandb.name = config.wandb.name + '_' + datetime.now().strftime("%Y%m%d%H%M%S")
    if config.get('wandb', None) is not None:
        wandb_logger = L.pytorch.loggers.WandbLogger(
            config=omegaconf.OmegaConf.to_object(config),
            ** config.wandb)

    if (config.checkpointing.resume_from_ckpt
            and config.checkpointing.resume_ckpt_path is not None
            and utils.fsspec_exists(
                config.checkpointing.resume_ckpt_path)):
        ckpt_path = config.checkpointing.resume_ckpt_path
    else:
        ckpt_path = None

    # Lightning callbacks
    callbacks = []
    if 'callbacks' in config:
        for _, callback in config.callbacks.items():
            callbacks.append(hydra.utils.instantiate(callback))

    train_ds, valid_ds = dataloader.get_dataloaders(
        config, tokenizer)
    _print_batch(train_ds, valid_ds, tokenizer)

    model = diffusion_model(config, tokenizer=valid_ds.tokenizer)
    if config.training.finetune_path != '':
        assert utils.fsspec_exists(config.training.finetune_path)
        logger.info(f'Finetuning from {config.training.finetune_path}')
        finetune_state_dict = torch.load(
            config.training.finetune_path,
            map_location='cpu', weights_only=False)
        if 'ema' in finetune_state_dict:
            ema_state_dict = finetune_state_dict['ema']
        if config.training.load_ema:
            finetune_state_dict = utils.convert_ema_state_dict(finetune_state_dict)
        else:
            finetune_state_dict = finetune_state_dict['state_dict']
            
        model.load_state_dict(
            finetune_state_dict,
            strict=False)
        model = model.cuda()
        model.metrics.to(model.device)
        
        if config.training.load_ema:
            ema_state_dict['shadow_params'] = [p.clone().to(model.device) for p in model._get_parameters()
                                               if p.requires_grad]
            model.ema.load_state_dict(ema_state_dict)
    if config.algo.name == 'im-sdtt':
        assert config.training.finetune_path != '', "IM-SDTT training requires a pretrained model checkpoint to finetune from."
        model.prepare_teacher_and_student()

    trainer = hydra.utils.instantiate(
        config.trainer,
        default_root_dir=config.checkpointing.save_dir,
        callbacks=callbacks,
        strategy=hydra.utils.instantiate(config.strategy),
        logger=wandb_logger)
    trainer.fit(model, train_ds, valid_ds, ckpt_path=ckpt_path)


@hydra.main(version_base=None, config_path='configs',
                        config_name='config')
def main(config):
    """Main entry point for training."""
    rank = int(os.getenv('LOCAL_RANK', '0'))
    L.seed_everything(config.seed + rank)
    _print_config(config, resolve=True, save_cfg=True)
    
    logger = utils.get_logger(__name__)
    tokenizer = dataloader.get_tokenizer(config)
    if config.algo.name == 'ar':
        diffusion_model = algo.AR
    elif config.algo.name == 'mdlm':
        diffusion_model = algo.MDLM
    elif config.algo.name == 'duo_base':
        diffusion_model = algo.DUO_BASE
    elif config.algo.name == 'd3pm':
        diffusion_model = algo.D3PMAbsorb
    elif config.algo.name == 'sedd':
        diffusion_model = algo.SEDDAbsorb
    elif config.algo.name == 'duo':
        diffusion_model = algo.DUO
    elif config.algo.name == 'distillation':
        diffusion_model = algo.Distillation
    elif config.algo.name == 'ot-finetune':
        diffusion_model = algo.OptimalTransportFinetune
    elif config.algo.name == 'rectification':
        diffusion_model = algo.Rectification
    elif config.algo.name == 'uniformbdd':
        diffusion_model = algo.UniformBDD
    elif config.algo.name == 'absorbbdd':
        diffusion_model = algo.AbsorbBDD
    elif config.algo.name == 'bdd':
        diffusion_model = algo.BDD
    elif config.algo.name == 'imdm':
        diffusion_model = algo.IMDM
    elif config.algo.name == 'im-redi':
        diffusion_model = algo.IMReDi
    elif config.algo.name == 'im-sdtt':
        diffusion_model = algo.IMSDTT
    else:
        raise ValueError(
            f'Invalid algorithm name: {config.algo.name}')
    kwargs = {'diffusion_model': diffusion_model,
                        'config': config,
                        'tokenizer': tokenizer,
                        'logger': logger}
    if config.mode == 'sample_eval':
        _generate_samples(**kwargs)
    elif config.mode == 'sample_eval_with_tc':
        _generate_samples_with_tc(**kwargs)
    elif config.mode == 'cond_sample_eval_with_tc':
        _generate_cond_samples_with_tc(**kwargs)
    elif config.mode == 'ppl_eval':
        _eval_ppl(**kwargs)
    elif config.mode == 'generate_reflow_dataset':
        generate_reflow_dataset(diffusion_model, config, logger, tokenizer)
    else:
        _train(**kwargs)


if __name__ == '__main__':
    # allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    main()
