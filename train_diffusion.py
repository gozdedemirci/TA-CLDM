
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,0"

from pathlib import Path
from datetime import datetime

import torch 
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import copy
from torch.utils import data

import sys
sys.path.append("..")
from src.data.datamodules import SimpleDataModule
from src.data.datasets import ROPDataset
from src.models.pipelines import DiffusionPipeline
from src.models.estimators import UNet
from src.models.noise_schedulers import GaussianNoiseScheduler
from src.models.embedders.time_embedder import TimeEmbbeding
from src.models.embedders.cond_embedders import LabelEmbedder
from src.models.embedders.latent_embedders import VAE

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


if __name__ == "__main__":
    # ------------ Load Data ----------------
   
    ds_train = ROPDataset(
        file_path='/home/gdemi/multi_view/multi_view_data', 
        split="train", 
        img_size=256
    )
    ds_val = ROPDataset(
        file_path='/home/gdemi/multi_view/multi_view_data', 
        split="val", 
        img_size=256
    )
    ds_test = ROPDataset(
        file_path='/home/gdemi/multi_view/multi_view_data', 
        split="test", 
        img_size=256
    )

  
    dm = SimpleDataModule(
        ds_train = ds_train,
        ds_val = ds_val,
        ds_test = ds_test,
        batch_size=16, 
        # num_workers=0,
        pin_memory=True,
        # weights=ds.get_weights()
    ) 
    
    current_time = datetime.now().strftime("%Y_%m_%d_%H%M%S")
    path_run_dir = Path.cwd() / '../experiments' / str(current_time)
    path_run_dir.mkdir(parents=True, exist_ok=True)
    accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'

    # ------------ Initialize Model ------------
    cond_embedder = LabelEmbedder 
    cond_embedder_kwargs = {
        'emb_dim': 1024,
        'num_classes': 2
    }
 
    time_embedder = TimeEmbbeding
    time_embedder_kwargs ={
        'emb_dim': 1024 # stable diffusion uses 4*model_channels (model_channels is about 256)
    }


    noise_estimator = UNet
    noise_estimator_kwargs = {
        'in_ch':8, 
        'out_ch':8, 
        'spatial_dims':2,
        'hid_chs':  [  256, 256, 512, 1024],
        'kernel_sizes':[3, 3, 3, 3],
        'strides':     [1, 2, 2, 2],
        'time_embedder':time_embedder,
        'time_embedder_kwargs': time_embedder_kwargs,
        'cond_embedder':cond_embedder,
        'cond_embedder_kwargs': cond_embedder_kwargs,
        'deep_supervision': False,
        'use_res_block':True,
        'use_attention':'none',
    }

    # ------------ Initialize Noise ------------
    noise_scheduler = GaussianNoiseScheduler
    noise_scheduler_kwargs = {
        'timesteps': 1000,
        'beta_start': 0.002, # 0.0001, 0.0015
        'beta_end': 0.02, # 0.01, 0.0195
        'schedule_strategy': 'scaled_linear'
    }
    
    # ------------ Initialize Latent Space  ------------
    latent_embedder = VAE(
        in_channels=3, 
        out_channels=3, 
        emb_channels=8,
        spatial_dims=2,
        hid_chs =    [ 64, 128, 256,  512], 
        kernel_sizes=[ 3,  3,   3,    3],
        strides =    [ 1,  2,   2,    2],
        deep_supervision=1,
        use_attention= 'none',
        loss = torch.nn.MSELoss,
        # optimizer_kwargs={'lr':1e-6},
        embedding_loss_weight=1e-6
    )
    latent_embedder_checkpoint = '/home/gdemi/multi_view/temp/experiments/latent_embedder_2025_02_04_183659/last.ckpt'
    cp = torch.load(latent_embedder_checkpoint)
    latent_embedder.load_state_dict(cp['state_dict'])
   
    # ------------ Initialize Pipeline ------------
    pipeline = DiffusionPipeline(
        noise_estimator=noise_estimator, 
        noise_estimator_kwargs=noise_estimator_kwargs,
        noise_scheduler=noise_scheduler, 
        noise_scheduler_kwargs = noise_scheduler_kwargs,
        latent_embedder=latent_embedder,
        latent_embedder_checkpoint = latent_embedder_checkpoint,
        estimator_objective='x_T',
        estimate_variance=False, 
        use_self_conditioning=False, 
        use_ema=False,
        classifier_free_guidance_dropout=0.5, # Disable during training by setting to 0
        do_input_centering=False,
        clip_x0=False,
        sample_every_n_steps=1000
    )
    
    # -------------- Training Initialization ---------------
    to_monitor = "train/loss"  # "pl/val_loss" 
    min_max = "min"
    save_and_sample_every = 100

    early_stopping = EarlyStopping(
        monitor=to_monitor,
        min_delta=0.0, # minimum change in the monitored quantity to qualify as an improvement
        patience=30, # number of checks with no improvement
        mode=min_max
    )
    checkpointing = ModelCheckpoint(
        dirpath=str(path_run_dir), # dirpath
        monitor=to_monitor,
        every_n_train_steps=save_and_sample_every,
        save_last=True,
        save_top_k=2,
        mode=min_max,
    )
    trainer = Trainer(
        accelerator=accelerator,
        default_root_dir=str(path_run_dir),
        callbacks=[checkpointing],
        check_val_every_n_epoch=1,
        log_every_n_steps=save_and_sample_every, 
        auto_lr_find=False,
        limit_val_batches=0, # 0 = disable validation - Note: Early Stopping no longer available 
        min_epochs=100,
        max_epochs=1001,
        num_sanity_val_steps=2,
        gpus=1
    )
    
    # ---------------- Execute Training ----------------
    trainer.fit(pipeline, datamodule=dm)

    # ------------- Save path to best model -------------
    pipeline.save_best_checkpoint(trainer.logger.log_dir, checkpointing.best_model_path)


