import numpy as np
import torch
import sys
import os
import pytorch_lightning as pl
from torch.backends import cudnn
from pathlib import Path
from pytorch_lightning.callbacks import ModelCheckpoint

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

from layout_data.utils.options import parses_ul
from layout_data.models.model import UnetUL


def main(hparams):
    """
    Main training routine specific for this project
    """
    seed = hparams.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.benchmark = True

    # ------------------------
    # 1 INIT LIGHTNING MODEL
    # ------------------------
    model = UnetUL(hparams)
    # ------------------------
    # 2 INIT TRAINER
    # ------------------------
    # Get SLURM job ID if available, otherwise use a default identifier
    job_id = os.environ.get('SLURM_JOB_ID', 'local')
    
    # Include job ID in checkpoint filename
    checkpoint_callback = ModelCheckpoint(
        save_top_k=1, 
        monitor='val_loss_jacobi', 
        mode='min',
        filename=f'job_{job_id}_noise{hparams.noise_std}_simple_{{epoch}}-{{step}}'
        #filename=f'job_{job_id}_ks{hparams.blur_kernel_size}_sg{hparams.blur_sigma}_complex_{{epoch}}-{{step}}'
    )
    trainer = pl.Trainer(
        max_epochs=hparams.max_epochs,
        callbacks=[checkpoint_callback],
        gpus=None if hparams.gpus == 0 else hparams.gpus,  # Use number of GPUs, not list
        precision=16 if hparams.use_16bit else 32,
        val_check_interval=hparams.val_check_interval,
        resume_from_checkpoint=hparams.resume_from_checkpoint,
        profiler=hparams.profiler,
        weights_summary=None,
        benchmark=True,
        logger=False,  # Disable logging to avoid disk quota issues
        default_root_dir='/work/chaabane/outputs/',  # Set work directory for all outputs
    )

    # ------------------------
    # 3 START TRAINING
    # ------------------------
    print(hparams)
    print()
    trainer.fit(model)


if __name__ == "__main__":
    # ------------------------
    # TRAINING ARGUMENTS
    # ------------------------
    config_path = Path(__file__).absolute().parent / "config_ul.yml"
    hparams = parses_ul(config_path)
    main(hparams)
