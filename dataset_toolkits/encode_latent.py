import os
import sys
import copy
import argparse
from contextlib import nullcontext
from concurrent.futures import ThreadPoolExecutor
from queue import Queue

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from easydict import EasyDict as edict

# Make UniLat3D importable when running as a script
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import unilat3d.models as models
import unilat3d.modules.sparse as sp


torch.set_grad_enabled(False)

# Hard-coded UniLat3D encoder weights (no CLI needed)
# This should be a valid `unilat3d.models.from_pretrained()` prefix:
# - local: <prefix>.json and <prefix>.safetensors exist
# - or HF path: "Org/Repo/path_prefix"
DEFAULT_UNILAT_ENCODER = "pretrained/encoder"


def _ensure_batched_coords(coords: np.ndarray) -> np.ndarray:
    """
    Ensure coords has shape (N, 4) with first column as batch index.
    Accepts (N, 3) or (N, 4).
    """
    if coords.ndim != 2 or coords.shape[1] not in (3, 4):
        raise ValueError(f"coords must have shape (N,3) or (N,4), got {coords.shape}")
    if coords.shape[1] == 4:
        return coords
    batch = np.zeros((coords.shape[0], 1), dtype=coords.dtype)
    return np.concatenate([batch, coords], axis=1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Keep CLI args consistent with TRELLIS/dataset_toolkits/encode_latent.py
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the metadata")
    parser.add_argument(
        "--filter_low_aesthetic_score",
        type=float,
        default=None,
        help="Filter objects with aesthetic score lower than this value",
    )
    parser.add_argument("--feat_model", type=str, default="dinov2_vitl14_reg", help="Feature model")
    parser.add_argument("--instances", type=str, default=None, help="Instances to process")
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world_size", type=int, default=1)
    # Extra (non-TRELLIS) but harmless: allows selecting cpu/cuda
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--use_autocast", action="store_true", default=True, help="Use autocast for mixed precision encoding (default: True)")
    parser.add_argument("--no_autocast", dest="use_autocast", action="store_false", help="Disable autocast")
    parser.add_argument("--autocast_dtype", type=str, default="bfloat16", choices=["bfloat16", "float16"], help="Autocast dtype (default: bfloat16)")

    opt = edict(vars(parser.parse_args()))

    # Build latent_name + load encoder (mirrors TRELLIS logic)
    device = torch.device(opt.device)
    latent_name = f"{opt.feat_model}_{DEFAULT_UNILAT_ENCODER.split('/')[-1]}"
    encoder = models.from_pretrained(DEFAULT_UNILAT_ENCODER).eval().to(device)
    
    # Parse autocast dtype
    autocast_dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }
    autocast_dtype = autocast_dtype_map.get(opt.autocast_dtype, torch.bfloat16)
    
    # Setup autocast context
    autocast_context = (
        torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=opt.use_autocast and device.type == "cuda")
        if opt.use_autocast and device.type == "cuda"
        else nullcontext()
    )
    
    if opt.use_autocast and device.type == "cuda":
        print(f"Using autocast with dtype={opt.autocast_dtype} for encoding")
    else:
        print("Autocast disabled, using full precision")

    save_dir = os.path.join(opt.output_dir, "latents", latent_name)
    os.makedirs(save_dir, exist_ok=True)

    meta_path = os.path.join(opt.output_dir, "metadata.csv")
    if not os.path.exists(meta_path):
        raise ValueError("metadata.csv not found")
    metadata = pd.read_csv(meta_path)

    if opt.instances is not None:
        with open(opt.instances, "r") as f:
            sha256s = [line.strip() for line in f if line.strip()]
        metadata = metadata[metadata["sha256"].isin(sha256s)]
    else:
        if opt.filter_low_aesthetic_score is not None:
            metadata = metadata[metadata["aesthetic_score"] >= opt.filter_low_aesthetic_score]
        metadata = metadata[metadata[f"feature_{opt.feat_model}"] == True]  # noqa: E712
        if f"latent_{latent_name}" in metadata.columns:
            metadata = metadata[metadata[f"latent_{latent_name}"] == False]  # noqa: E712

    # Shard by rank/world_size
    start = len(metadata) * opt.rank // opt.world_size
    end = len(metadata) * (opt.rank + 1) // opt.world_size
    metadata = metadata[start:end]

    records = []

    # Filter out already-saved files (resume-friendly)
    sha256s = list(metadata["sha256"].astype(str).values)
    for sha256 in copy.copy(sha256s):
        save_path = os.path.join(save_dir, f"{sha256}.npz")
        if os.path.exists(save_path):
            records.append({"sha256": sha256, f"latent_{latent_name}": True})
            sha256s.remove(sha256)

    load_queue: "Queue[tuple[str, Optional[np.ndarray], Optional[np.ndarray]]]" = Queue(maxsize=4)

    def loader(sha256: str) -> None:
        try:
            in_path = os.path.join(opt.output_dir, "features", opt.feat_model, f"{sha256}.npz")
            with np.load(in_path, allow_pickle=False) as pack:
                feats = pack["patchtokens"]
                coords = pack["indices"]
            load_queue.put((sha256, feats, coords))
        except Exception as e:
            print(f"[encode_latent] Error loading features for {sha256}: {e}")
            load_queue.put((sha256, None, None))

    def saver(sha256: str, out_pack: dict) -> None:
        save_path = os.path.join(save_dir, f"{sha256}.npz")
        np.savez_compressed(save_path, **out_pack)
        records.append({"sha256": sha256, f"latent_{latent_name}": True})

    try:
        with ThreadPoolExecutor(max_workers=32) as loader_executor, ThreadPoolExecutor(max_workers=32) as saver_executor:
            loader_executor.map(loader, sha256s)

            for _ in tqdm(range(len(sha256s)), desc="Extracting latents"):
                sha256, feats, coords = load_queue.get()
                if feats is None or coords is None:
                    continue
                try:
                    coords = _ensure_batched_coords(coords)

                    x = sp.SparseTensor(
                        feats=torch.from_numpy(feats).float(),
                        coords=torch.from_numpy(coords).to(torch.int32),
                    ).to(device)

                    # NOTE: UniLatEncoder.forward(return_raw=True) expects sample_posterior=True (otherwise `std` may be undefined).
                    # Use autocast for mixed precision encoding
                    with autocast_context:
                        z, mean, logvar, _, _ = encoder(x, sample_posterior=True, return_raw=True)

                    # Convert to float32 before converting to numpy (numpy doesn't support bfloat16)
                    # This ensures compatibility even when autocast produces bfloat16 tensors
                    mean = mean.detach().cpu().float()
                    logvar = logvar.detach().cpu().float()
                    z = z.detach().cpu().float()

                    assert torch.isfinite(mean).all(), "Non-finite latent mean"
                    assert torch.isfinite(logvar).all(), "Non-finite latent logvar"

                    out_pack = {
                        "mean": mean.numpy().astype(np.float32),
                        "logvar": logvar.numpy().astype(np.float32),
                        "sample": z.numpy().astype(np.float32),
                    }
                    saver_executor.submit(saver, sha256, out_pack)
                except Exception as e:
                    print(f"[encode_latent] Error encoding {sha256}: {e}")

            saver_executor.shutdown(wait=True)
    except Exception as e:
        print(f"[encode_latent] Error happened during processing: {e}")

    # Write shard record csv (mirrors TRELLIS style)
    records_df = pd.DataFrame.from_records(records)
    out_csv = os.path.join(opt.output_dir, f"latent_{latent_name}_{opt.rank}.csv")
    records_df.to_csv(out_csv, index=False)


