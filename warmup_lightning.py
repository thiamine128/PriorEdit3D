import os
import sys
import json
import glob
import argparse
from easydict import EasyDict as edict

import torch
import numpy as np

from unilat3d import models, datasets

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from unilat3d.lightning import UniLatDataModule, UnpairedLightningModule, WarmupLightningModule


def find_ckpt(cfg):
    cfg["load_ckpt_path"] = None
    if cfg.load_dir != "":
        if cfg.ckpt == "latest":
            ckpt_dir = os.path.join(cfg.load_dir, "lightning_ckpts")
            last_ckpt = os.path.join(ckpt_dir, "last.ckpt")
            if os.path.exists(last_ckpt):
                cfg["load_ckpt_path"] = last_ckpt
            else:
                ckpts = sorted(glob.glob(os.path.join(ckpt_dir, "*.ckpt")), key=os.path.getmtime)
                cfg["load_ckpt_path"] = ckpts[-1] if ckpts else None
        elif cfg.ckpt == "none":
            cfg["load_ckpt_path"] = None
        else:
            ckpt_dir = os.path.join(cfg.load_dir, "lightning_ckpts")
            step = int(cfg.ckpt)
            matches = sorted(glob.glob(os.path.join(ckpt_dir, f"*step={step}*.ckpt")))
            cfg["load_ckpt_path"] = matches[0] if matches else None
    return cfg


def main(cfg):
    seed = int(getattr(cfg, "seed", 0))
    pl.seed_everything(seed, workers=True)
    cfg = find_ckpt(cfg)
    dataset = getattr(datasets, cfg.dataset.name)(cfg.data_dir, **cfg.dataset.args)
    cfg.models['denoiser'].args['precision'] = 'fp32'
    model = getattr(models, cfg.models['denoiser'].name)(**cfg.models['denoiser'].args)
    targs = cfg.trainer.args
    module = WarmupLightningModule(
        model,
        dataset,
        output_dir=cfg.output_dir,
        **targs,
    )

    init_ckpt = str(getattr(cfg, "init_ckpt", "") or "")
    if init_ckpt:
        ckpt_obj = torch.load(init_ckpt, map_location="cpu")
        state_dict = ckpt_obj.get("state_dict", ckpt_obj) if isinstance(ckpt_obj, dict) else ckpt_obj
        strict = bool(getattr(cfg, "init_ckpt_strict", False))
        state_dict = {k.replace("models.denoiser.", "model."): v for k, v in state_dict.items()}
        missing, unexpected = module.load_state_dict(state_dict, strict=strict)
        print(f"[init_ckpt] loaded: {init_ckpt}")
        print(f"[init_ckpt] strict={strict}, missing={len(missing)}, unexpected={len(unexpected)}")

    default_workers = int(np.ceil(os.cpu_count() / max(1, torch.cuda.device_count())))
    num_workers = int(getattr(dataset, "num_workers", default_workers))

    dm = UniLatDataModule(dataset, batch_size=1, num_workers=num_workers, shuffle=True)
    logger = TensorBoardLogger(save_dir=cfg.output_dir, name="tb_logs", version="")

    ckpt_dir = os.path.join(cfg.output_dir, "lightning_ckpts")
    ckpt_every = 100
    checkpoint_cb = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="step={step}",
        save_last=True,
        save_top_k=-1,
        every_n_train_steps=ckpt_every if ckpt_every > 0 else None,
        save_on_train_epoch_end=False,
    )

    callbacks = [
        checkpoint_cb,
    ]

    trainer = pl.Trainer(
        default_root_dir=cfg.output_dir,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=int(cfg.num_gpus) if torch.cuda.is_available() else 1,
        num_nodes=int(cfg.num_nodes),
        max_steps=int(targs["max_steps"]),
        logger=logger,
        callbacks=callbacks,
        log_every_n_steps=max(1, int(targs.get("i_log", 50))),
        enable_checkpointing=True,
        precision="16-mixed",
    )

    trainer.fit(module, datamodule=dm, ckpt_path=cfg.get("load_ckpt_path", None))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # config
    parser.add_argument("--config", type=str, required=True, help="Experiment config file (json)")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--load_dir", type=str, default="", help="Load directory, default to output_dir")
    parser.add_argument("--ckpt", type=str, default="latest", help="Checkpoint step to resume training, default to latest")
    parser.add_argument("--data_dir", type=str, default="./data/", help="Data directory")
    parser.add_argument("--num_nodes", type=int, default=1, help="Number of nodes")
    parser.add_argument("--node_rank", type=int, default=0, help="Node rank (kept for compatibility; Lightning handles ranks)")
    parser.add_argument("--num_gpus", type=int, default=-1, help="Number of GPUs per node, default to all")
    parser.add_argument(
        "--init_ckpt",
        type=str,
        default="",
        help="Optional weights-only Lightning checkpoint (.ckpt) to initialize model weights before training.",
    )
    parser.add_argument(
        "--init_ckpt_strict",
        action="store_true",
        help="Use strict=True when loading --init_ckpt (default: strict=False).",
    )
    opt = parser.parse_args()

    opt.load_dir = opt.load_dir if opt.load_dir != "" else opt.output_dir
    opt.num_gpus = torch.cuda.device_count() if opt.num_gpus == -1 else opt.num_gpus

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

