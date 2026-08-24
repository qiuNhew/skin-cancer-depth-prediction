#!/usr/bin/env python
"""V4 multi-task training entry point — orchestrates dataloaders, multi-task model,
multi-task loss, optimiser, MultiTaskTrainer.

Identical CLI / behaviour to `train.py` except it instantiates the multi-task
trainer that handles the model's dict outputs and the depth target. Kept as a
parallel script (rather than a flag on train.py) so V1/V2/V3 training is
guaranteed to be untouched.

Usage:
    python breslow_depth_prediction/scripts/train_multitask.py
    python breslow_depth_prediction/scripts/train_multitask.py --config breslow_depth_prediction/configs/config_v4.yaml
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# Add project root so `breslow_depth_prediction.*` imports work.
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from breslow_depth_prediction.src.config import load_config, resolve_paths, save_config
from breslow_depth_prediction.src.data import create_dataloaders, set_seed
from breslow_depth_prediction.src.models import get_model, get_loss_function
from breslow_depth_prediction.src.training import MultiTaskTrainer
from breslow_depth_prediction.src.visualization import plot_training_history, save_figure


def setup_logging(log_dir: Path) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"train_multitask_{time.strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file),
        ],
    )
    logger = logging.getLogger(__name__)
    logger.info(f"Logging to {log_file}")
    return logger


def main():
    parser = argparse.ArgumentParser(description="Train V4 multi-task Breslow model")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to configuration file (default: configs/config_v4.yaml)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume training from")
    parser.add_argument("--checkpoint-name", type=str, default="best_model.pth",
                        help="Filename to save the best checkpoint as (default: best_model.pth).")
    args = parser.parse_args()

    config_path = (
        Path(args.config) if args.config is not None
        else project_root / "breslow_depth_prediction" / "configs" / "config_v4.yaml"
    )
    if not config_path.exists():
        print(f"ERROR: Config file not found: {config_path}")
        sys.exit(1)

    config = load_config(str(config_path))
    runtime_config = resolve_paths(config, project_root)

    paths_config = runtime_config.get("paths", {})
    checkpoint_dir = Path(paths_config.get("checkpoint_dir", project_root / "checkpoints"))
    results_dir = Path(paths_config.get("results_dir", project_root / "results"))
    log_dir = Path(paths_config.get("log_dir", project_root / "logs"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(log_dir)
    logger.info(f"Project root: {project_root}")
    logger.info(f"Config: {config_path}")
    logger.info(f"Checkpoint name: {args.checkpoint_name}")

    seed = runtime_config.get("split", {}).get("random_seed", 42)
    set_seed(seed)
    logger.info(f"Random seed: {seed}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")
    if device == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info(f"VRAM: {vram_gb:.1f} GB")
        torch.backends.cudnn.benchmark = True

    # Write V4 snapshot under a versioned name so V3's config_used.yaml is preserved.
    save_config(config, results_dir / "config_used_v4.yaml")

    # === Data ============================================================
    logger.info("Creating dataloaders...")
    train_loader, val_loader, test_loader = create_dataloaders(runtime_config)
    logger.info(
        f"Dataset: train={len(train_loader.dataset)}, "
        f"val={len(val_loader.dataset)}, test={len(test_loader.dataset)}"
    )

    # === Model ===========================================================
    logger.info("Creating model...")
    model = get_model(runtime_config)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")

    # === Loss / optimiser / scheduler ====================================
    criterion = get_loss_function(runtime_config)  # MultiTaskLoss

    training_config = runtime_config.get("training", {})
    lr = training_config.get("learning_rate", 1e-4)
    weight_decay = training_config.get("weight_decay", 1e-4)
    num_epochs = training_config.get("num_epochs", 100)
    patience = training_config.get("early_stopping_patience", 15)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

    logger.info(f"Optimizer: AdamW (lr={lr}, weight_decay={weight_decay})")
    logger.info(f"Scheduler: CosineAnnealingLR (T_max={num_epochs})")
    logger.info(f"Epochs: {num_epochs}, Early stopping patience: {patience}")

    # === Trainer =========================================================
    num_classes = runtime_config.get("classes", {}).get("num_classes", 4)
    trainer = MultiTaskTrainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        scheduler=scheduler,
        checkpoint_dir=str(checkpoint_dir),
        mixed_precision=False,
        num_classes=num_classes,
    )

    start_epoch = 0
    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.exists():
            logger.error(f"Checkpoint not found: {resume_path}")
            sys.exit(1)
        start_epoch = trainer.load_checkpoint(str(resume_path))
        logger.info(f"Resumed from checkpoint at epoch {start_epoch}")

    # The base Trainer.fit always saves to "best_model.pth"; redirect by
    # monkey-patching `save_checkpoint` so this run doesn't clobber V3.
    target_filename = args.checkpoint_name
    _orig_save = trainer.save_checkpoint
    def _redirected_save(epoch, filename):
        # Only intercept the "best model" save; checkpoint_epoch_X stays as-is.
        if filename == "best_model.pth":
            return _orig_save(epoch, target_filename)
        return _orig_save(epoch, filename)
    trainer.save_checkpoint = _redirected_save

    logger.info("Starting training...")
    start_time = time.time()
    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=num_epochs,
        early_stopping_patience=patience,
        save_best_only=True,
    )
    elapsed = time.time() - start_time
    logger.info(f"Training complete in {elapsed / 60:.1f} minutes")
    logger.info(f"Best validation Dice: {trainer.best_val_dice:.4f}")
    logger.info(f"Best validation Loss: {trainer.best_val_loss:.4f}")
    logger.info(f"Best model saved as: {target_filename}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        fig = plot_training_history(history)
        save_figure(fig, str(results_dir / "training_history_v4.png"))
        logger.info(f"Saved training history plot to {results_dir / 'training_history_v4.png'}")
    except Exception as e:
        logger.warning(f"Could not save training history plot: {e}")

    with open(results_dir / "training_summary_v4.txt", "w") as f:
        f.write("Training Summary (V4 multi-task)\n")
        f.write(f"{'='*50}\n")
        f.write(f"Model: {runtime_config['model'].get('architecture')}\n")
        f.write(f"Encoder: {runtime_config['model'].get('encoder')}\n")
        f.write(f"Epochs trained: {len(history['train_loss'])}\n")
        f.write(f"Best val Dice: {trainer.best_val_dice:.4f}\n")
        f.write(f"Best val Loss: {trainer.best_val_loss:.4f}\n")
        f.write(f"Training time: {elapsed / 60:.1f} minutes\n")
        f.write(f"Device: {device}\n")
        f.write(f"Checkpoint: {checkpoint_dir / target_filename}\n")

    logger.info("All results saved. Training complete!")


if __name__ == "__main__":
    main()
