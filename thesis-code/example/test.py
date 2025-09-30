"""
Runs a model on a single node across multiple gpus.
"""
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.backends import cudnn
import sys
import os
import scipy.io as sio
import glob
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

from layout_data.loss.ulloss import Jacobi_layer
from layout_data.models.model import UnetUL
from layout_data.utils.options import parses_ul

def main(hparams):
    """
    Main training routine specific for this project
    """
    seed = hparams.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.benchmark = True

    model = UnetUL(hparams).cuda()

    # Update according to the case - you can manually specify the checkpoint path
    model_path = '/work/chaabane/outputs/checkpoints/job_5564520_noise0.0003_simple_epoch=29-step=239999.ckpt' # Manual path
    print(f"Loading model from: {model_path}")
    model.load_state_dict(torch.load(model_path)['state_dict'])
    model.prepare_data()
    data_loader = model.test_dataloader()
    mae_list, cmae_list, maxae_list, mtae_list, loss_list = [], [], [], [], []
    jacobi = Jacobi_layer(nx=200, length=0.1, bcs=[[[0.045, 0.0], [0.055, 0.0]]])
    for idx, data in enumerate(data_loader):
        model.eval()
        layout, heat = data[0].cuda(), data[1].cuda()
        with torch.no_grad():
            heat_pre = model(layout)

            # Loss
            loss = F.l1_loss(heat_pre, jacobi(layout, heat_pre.detach(), 1))
            loss_list.append(loss.item())

            heat_pre = heat_pre + 298
            # MAE
            mae = F.l1_loss(heat, heat_pre)
            mae_list.append(mae.item())
            # CMAE
            cmae = F.l1_loss(heat, heat_pre, reduction='none').squeeze()
            x_index, y_index = torch.where(layout.squeeze() > 0.0)
            cmae = cmae[x_index, y_index]
            cmae = cmae.mean()
            cmae_list.append(cmae.item())
            # Max AE
            maxae = F.l1_loss(heat, heat_pre, reduction='none').squeeze()
            maxae = torch.max(maxae)
            maxae_list.append(maxae.item())
            # MT AE
            mtae = torch.abs(torch.max(heat_pre) - torch.max(heat))
            mtae_list.append(mtae.item())

        if (idx + 1) % 500 == 0:
            print(idx + 1)

    loss_list = np.array(loss_list)
    mae_list = np.array(mae_list)
    cmae_list = np.array(cmae_list)
    maxae_list = np.array(maxae_list)
    mtae_list = np.array(mtae_list)
    print('-' * 20)
    print('Loss:', loss_list.mean())
    print('MAE:', mae_list.mean())
    print('CMAE:', cmae_list.mean())
    print('Max AE:', maxae_list.mean())
    print('MT AE:', mtae_list.mean())
    noise_std = getattr(hparams, 'noise_std', 0.0)
    blur_kernel_size = getattr(hparams, 'blur_kernel_size', 5)
    blur_sigma = getattr(hparams, 'blur_sigma', 1.0)
    seed = getattr(hparams, 'seed', 34 )
    #output_name = f"outputs/pi_unet_testing_simple_augmentation/test_results_simple_ks_5_sg0.5.mat"
    #output_name = f"outputs/pi_unet_testing_complex_augmentation/test_results_complex_ks_5_sg0.8.mat"
    #output_name = f"outputs/pi_unet_testing_simple_clean_on_clean/test_results_simple_seed11.mat"
    output_name = f"/homes/math/chaabane/code/PI-UNet_HSL-TFP/example/outputs/heatmap_simple/test_results_simple_GN_on_GN.mat"
    #output_name = f"/work/chaabane/outputs/pi_unet_testing_simple_noise/test_results_simple_noise0.0003.mat"
    sio.savemat(output_name,
                {'loss': loss_list, 'mae': mae_list, 'cmae': cmae_list, 'maxae': maxae_list, 'mtae': mtae_list})


if __name__ == "__main__":
    # ------------------------
    # TRAINING ARGUMENTS
    # ------------------------
    config_path = Path(__file__).absolute().parent / "config_ul.yml"
    hparams = parses_ul(config_path)

    # --- Ensure these are always set, regardless of CLI/YAML ---
    hparams.noise_std = getattr(hparams, 'noise_std', 0.0)         # or set to your desired default
    hparams.blur_sigma = getattr(hparams, 'blur_sigma', 0.5)       # or set to your desired default
    hparams.blur_kernel_size = getattr(hparams, 'blur_kernel_size', 3) # or set to your desired default
    # -----------------------------------------------------------

    main(hparams)
