#!/usr/bin/env python3
"""复用已有的预训练权重，重跑下游分类与特征可视化

预训练（10 epoch）在 CPU 上较慢。若已有 checkpoint，可用本脚本跳过预训练，
只重跑分类头训练与可视化。

用法:
    python scripts/rerun_classification.py
    python scripts/rerun_classification.py --experiment-dir results/experiment_20260624_151338
"""
import argparse
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from kmer_encoding import KmerEncoder                              # noqa: E402
from masked_reconstruction_model import MaskedReconstructionModel  # noqa: E402
from experiment_runner import ExperimentRunner                     # noqa: E402

MODEL_KWARGS = dict(
    embedding_dim=128,
    num_filters=[128, 256, 512],
    kernel_sizes=[3, 3, 3],
    encoder_output_dim=256,
    mask_ratio=0.15,
    dropout=0.1,
)


def main() -> None:
    ap = argparse.ArgumentParser(description="跳过预训练，重跑下游分类")
    ap.add_argument(
        "--experiment-dir",
        default="results/experiment_20260624_151338",
        help="包含 pretraining/final_model.pt 的实验目录",
    )
    ap.add_argument("--classification-epochs", type=int, default=50)
    ap.add_argument("--classification-lr", type=float, default=1e-3)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    exp_dir = (PROJECT_ROOT / args.experiment_dir).resolve()
    results_json = exp_dir / "experiment_results.json"
    ckpt_path = exp_dir / "pretraining" / "final_model.pt"

    if not ckpt_path.exists():
        raise SystemExit(f"未找到预训练权重: {ckpt_path}")

    # 本次不重跑预训练，保留原有记录
    pretraining = {}
    if results_json.exists():
        pretraining = json.loads(results_json.read_text(encoding="utf-8")).get(
            "pretraining", {}
        )

    runner = ExperimentRunner(
        data_dir=str(PROJECT_ROOT / "data"),
        results_dir=str(exp_dir),
        num_epochs=10,  # 沿用原始预训练轮数，仅用于记录
        batch_size=32,
        learning_rate=1e-4,
        classification_epochs=args.classification_epochs,
        classification_lr=args.classification_lr,
        device=args.device,
        max_length=args.max_length,
        k=args.k,
        mask_ratio=0.15,
    )

    # 重建预训练模型并载入权重
    kmer = KmerEncoder(k=args.k, max_length=args.max_length)
    vocab_size = len(kmer.vocabulary)
    model = MaskedReconstructionModel(
        vocab_size=vocab_size, reconstruction_dim=vocab_size, **MODEL_KWARGS
    )
    ckpt = torch.load(ckpt_path, map_location=args.device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[checkpoint] 缺失键 {len(missing)} / 多余键 {len(unexpected)}")
    runner.model = model

    runner.run_classification()
    runner.generate_visualizations()

    runner.pretraining_results = pretraining
    runner.collect_metrics()
    runner.save_results()

    m = runner.classification_results["metrics"]
    print()
    print("=" * 60)
    print("下游分类结果（验证集）")
    print("=" * 60)
    print(f"accuracy  = {m['accuracy']:.4f}")
    print(f"precision = {m['precision']:.4f}")
    print(f"recall    = {m['recall']:.4f}")
    print(f"f1        = {m['f1']:.4f}")
    print(f"最佳 val_acc = {runner.classification_results['best_val_accuracy']:.4f}")
    print(f"结果已写入 {results_json}")


if __name__ == "__main__":
    main()
