import argparse
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


class ResNet(nn.Module):
    def __init__(self, hash_bit: int):
        super().__init__()
        model_resnet = models.resnet50(pretrained=False)
        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4
        self.avgpool = model_resnet.avgpool
        self.feature_layers = nn.Sequential(
            self.conv1,
            self.bn1,
            self.relu,
            self.maxpool,
            self.layer1,
            self.layer2,
            self.layer3,
            self.layer4,
            self.avgpool,
        )

        self.hash_layer = nn.Linear(model_resnet.fc.in_features, hash_bit)
        self.hash_layer.weight.data.normal_(0, 0.01)
        self.hash_layer.bias.data.fill_(0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_layers(x)
        x = x.view(x.size(0), -1)
        y = self.hash_layer(x)
        return y


def _default_repo_root() -> Path:
    # scripts/ -> repo root
    return Path(__file__).resolve().parents[1]


def _load_image(path: Path, transform) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    return transform(img)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Precompute trn_binary.npy for the Flask demo (ImageNet database split)."
    )
    parser.add_argument(
        "--imagenet-root",
        type=Path,
        default=None,
        help="Directory that contains the paths referenced in database.txt (e.g. has image/ subfolder). "
        "Defaults to $IMAGENET_ROOT or <repo>/data/imagenet.",
    )
    parser.add_argument(
        "--database-txt",
        type=Path,
        default=None,
        help="Path to database.txt. Defaults to <repo>/data/imagenet/database.txt.",
    )
    parser.add_argument(
        "--model-pt",
        type=Path,
        required=True,
        help="Path to model.pt (state_dict saved by this repo).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Output directory where trn_binary.npy will be written.",
    )
    parser.add_argument(
        "--bit",
        type=int,
        default=64,
        help="Hash length (must match the model). Default: 64.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for inference. Default: 64.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help='Torch device, e.g. "cpu" or "mps". Default: cpu.',
    )
    parser.add_argument(
        "--skip-missing",
        action="store_true",
        help="Skip images that are missing/corrupt and also write database.filtered.txt aligned to trn_binary.npy.",
    )
    args = parser.parse_args()

    repo_root = _default_repo_root()

    database_txt = args.database_txt or (repo_root / "data" / "imagenet" / "database.txt")
    if not database_txt.exists():
        raise FileNotFoundError(f"database.txt not found: {database_txt}")

    imagenet_root = args.imagenet_root or Path(
        os.environ.get("IMAGENET_ROOT", str(repo_root / "data" / "imagenet"))
    )

    device = torch.device(args.device)

    tfm = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    print(f"Using database.txt: {database_txt}")
    print(f"Using IMAGENET_ROOT: {imagenet_root}")
    print(f"Loading model: {args.model_pt}")
    print(f"Device: {device}")

    net = ResNet(args.bit).to(device)
    state_dict = torch.load(args.model_pt, map_location=device)
    net.load_state_dict(state_dict)
    net.eval()

    rel_paths: list[str] = []
    abs_paths: list[Path] = []
    with database_txt.open("r") as f:
        for line in f:
            if not line.strip():
                continue
            rel = line.split(" ")[0]
            rel_paths.append(rel)
            abs_paths.append(imagenet_root / rel)

    if not rel_paths:
        raise RuntimeError(f"No entries found in {database_txt}")

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    all_codes: list[np.ndarray] = []
    kept_rel_paths: list[str] = []

    with torch.no_grad():
        batch_imgs: list[torch.Tensor] = []
        batch_rels: list[str] = []

        def flush_batch():
            if not batch_imgs:
                return
            x = torch.stack(batch_imgs, dim=0).to(device)
            y = net(x)
            codes = y.sign().cpu().numpy().astype(np.int8)
            all_codes.append(codes)
            kept_rel_paths.extend(batch_rels)
            batch_imgs.clear()
            batch_rels.clear()

        for rel, p in zip(rel_paths, abs_paths):
            try:
                if not p.exists():
                    raise FileNotFoundError(str(p))
                img_t = _load_image(p, tfm)
            except Exception as e:
                if args.skip_missing:
                    print(f"[skip] {rel}: {e}")
                    continue
                raise FileNotFoundError(
                    f"Missing/corrupt image referenced by database.txt:\n- rel: {rel}\n- abs: {p}\n\n"
                    "Fix your IMAGENET_ROOT, or rerun with --skip-missing."
                ) from e

            batch_imgs.append(img_t)
            batch_rels.append(rel)
            if len(batch_imgs) >= args.batch_size:
                flush_batch()

        flush_batch()

    trn_binary = np.concatenate(all_codes, axis=0)
    np.save(out_dir / "trn_binary.npy", trn_binary)
    print(f"Saved: {out_dir / 'trn_binary.npy'}  shape={trn_binary.shape} dtype={trn_binary.dtype}")

    if args.skip_missing:
        filtered = out_dir / "database.filtered.txt"
        with filtered.open("w") as f:
            for rel in kept_rel_paths:
                f.write(rel + "\n")
        print(f"Saved: {filtered} (aligned with trn_binary.npy)")
        print(
            "Note: update the Flask demo to read database.filtered.txt if you use --skip-missing."
        )


if __name__ == "__main__":
    main()

