import os
import sys
import json
import glob
import argparse
from easydict import EasyDict as edict
from torch.distributed.fsdp.api import MixedPrecision

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

from pytorch_lightning.strategies import DeepSpeedStrategy, FSDPStrategy
import torch
import numpy as np

from unilat3d import models, datasets

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from unilat3d.lightning import UniLatDataModule, UnpairedLightningModule, WarmupLightningModule

os.environ['NCCL_TIMEOUT'] = '7200'

def load_ckpt_state_dict(path: str) -> dict:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    return obj["state_dict"]

def extract_denoiser_state_dict(prefix: str, lightning_state_dict: dict) -> dict:
    out = {k[len(prefix) :]: v for k, v in lightning_state_dict.items() if k.startswith(prefix)}
    return out


def main(cfg):
    seed = int(getattr(cfg, "seed", 0))
    pl.seed_everything(seed, workers=True)
    dataset = getattr(datasets, cfg.dataset.name)(cfg.data_dir, **cfg.dataset.args)
    model_dict = {name: getattr(models, m.name)(**m.args) for name, m in cfg.models.items()}
    targs = dict(cfg.trainer.args)
    edit_model_ckpt = cfg.edit_model_ckpt
    aux_model_ckpt = cfg.aux_model_ckpt
    edit_model_state_dict = extract_denoiser_state_dict("model.", load_ckpt_state_dict(edit_model_ckpt))
    model_dict["denoiser"].load_state_dict(edit_model_state_dict)
    aux_model_state_dict = extract_denoiser_state_dict("model.", load_ckpt_state_dict(aux_model_ckpt))
    model_dict["auxiliary_model"].load_state_dict(aux_model_state_dict, strict=False)
    module = UnpairedLightningModule(
            model_dict["denoiser"],
            model_dict["auxiliary_model"],
            output_dir=cfg.output_dir,
            cam=dataset.cam,
            vlm_cam=dataset.vlm_cam,
            consistency_cam=dataset.consistency_cam,
            **targs,
        )

    default_workers = int(np.ceil(os.cpu_count() / max(1, torch.cuda.device_count())))
    num_workers = int(getattr(dataset, "num_workers", default_workers))


    dm = UniLatDataModule(
        dataset,
        batch_size=1,
        num_workers=num_workers,
        shuffle=True,
        seed=seed,
        exact_resume=True,
        use_distributed_sampler=True,
    )
    logger = TensorBoardLogger(save_dir=cfg.output_dir, name="tb_logs", version="")
    ckpt_dir = os.path.join(cfg.output_dir, "lightning_ckpts")
    ckpt_every = cfg.ckpt_every
    checkpoint_cb = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="step={step}",
        save_last=True,
        save_top_k=-1,
        save_weights_only=False,  # required for resume (optimizer/scheduler state)
        every_n_train_steps=ckpt_every if ckpt_every > 0 else None,
        save_on_train_epoch_end=False,
    )

    callbacks = [
        checkpoint_cb,
    ]

    trainer = pl.Trainer(
        default_root_dir=cfg.output_dir,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices='auto',
        logger=logger,
        precision="bf16",
        strategy=DeepSpeedStrategy(config="configs/deepspeed_config.json"),
        use_distributed_sampler=False,
        log_every_n_steps=1,
        callbacks=callbacks,
        max_epochs=-1,
    )

    ckpt_path = getattr(cfg, "resume_from", None)
    if ckpt_path is None:
        ckpt_dir = os.path.join(cfg.output_dir, "lightning_ckpts")
        last_ckpt = os.path.join(ckpt_dir, "last.ckpt")
        if os.path.exists(last_ckpt):
            ckpt_path = last_ckpt
            print(f"Found existing checkpoint: {ckpt_path}, resuming training...", flush=True)
    if ckpt_path is None:
        print("No resume checkpoint found, starting from scratch.", flush=True)
    else:
        print(f"Using ckpt_path for trainer.fit: {ckpt_path}", flush=True)

    trainer.fit(module, datamodule=dm, ckpt_path=ckpt_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # config
    parser.add_argument("--config", type=str, required=True, help="Experiment config file (json)")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--data_dir", type=str, default="./data/", help="Data directory")
    parser.add_argument("--resume_from", type=str, default=None, help="Path to checkpoint to resume from. If not provided, will automatically look for 'last.ckpt' in output_dir/lightning_ckpts/")
    parser.add_argument("--edit_model_ckpt", type=str, default=None, help="Path to edit model checkpoint")
    parser.add_argument("--aux_model_ckpt", type=str, default=None, help="Path to auxiliary model checkpoint")
    opt = parser.parse_args()

    config = json.load(open(opt.config, "r"))

    cfg = edict()
    cfg.update(opt.__dict__)
    cfg.update(config)

    os.makedirs(cfg.output_dir, exist_ok=True)
    with open(os.path.join(cfg.output_dir, "command.txt"), "w") as fp:
        print(" ".join(["python"] + sys.argv), file=fp)
    with open(os.path.join(cfg.output_dir, "config.json"), "w") as fp:
        json.dump(config, fp, indent=4)

    main(cfg)

