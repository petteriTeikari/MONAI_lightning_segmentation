# Minimum working example for trying to make 3rd party networks to work with MONAI dataset + transformations
# Petteri Teikari, July 2020
# modified from:
# https://github.com/Project-MONAI/MONAI/blob/master/examples/notebooks/spleen_segmentation_3d_lightning.ipynb

#
import os
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

import monai
from monai.transforms import \
    Compose, LoadNiftid, AddChanneld, ScaleIntensityRanged, RandCropByPosNegLabeld, \
    CropForegroundd, Rand3DElasticd, RandGaussianNoised, Spacingd, Orientationd, ToTensord, Rotated
from monai.data import list_data_collate
from monai.inferers import sliding_window_inference
from monai.networks.layers import Norm
from monai.metrics import compute_meandice
from monai.utils import set_determinism
from pytorch_lightning import LightningModule, Trainer, loggers
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateLogger, EarlyStopping

from model_unet_attention import unet_CT_multi_att_dsv_3D, unet_grid_attention_3D

monai.config.print_config()
print('NVIDIA Driver version (well non-torch CUDA version) = {}'.format(torch._C._cuda_getDriverVersion()))
print('CUDNN version = {} (e.g. 7605 -> 7.6.05 (major.minor.patchlevel)'.format(torch.backends.cudnn.version()))
print('CUDA version = {}\n'.format(torch.version.cuda))

class Net(LightningModule):
    def __init__(self):
        super().__init__()
        if_3rdparty = True
        attention_variant = 'unet_grid_attention_3D'
        if if_3rdparty:
            # Use 3rd party Pytorch network
            print('Training using the 3rd Party model "U-Net with Attention"')
            print(' using the variant = {}'.format(attention_variant))
            if attention_variant == 'unet_grid_attention_3D':
                self._model = unet_grid_attention_3D(n_classes=2,
                                                     in_channels=1,
                                                     attention_dsample=(2, 2, 2),  # default (2,2,2)
                                                     filters=[4, 8, 16, 32, 64])  # divided by scale = 4
                                                     # -> corresponding to channels = [3, 6, 12, 24, 48]
            elif attention_variant == 'unet_CT_multi_att_dsv_3D':
                self._model = unet_CT_multi_att_dsv_3D(n_classes=2,
                                                       in_channels=1,
                                                       attention_dsample = (2,2,2), # default (2,2,2)
                                                       filters=[4, 8, 16, 32, 64]) # divided by scale = 4
                                                       # -> corresponding to channels = [3, 6, 12, 24, 48]
        else:
            # Use MONAI off-the-shelf network
            print('Training using the standard MONAI U-Net')
            self._model = monai.networks.nets.UNet(dimensions=3, in_channels=1, out_channels=2,
                                                   channels=(16, 32, 64, 128, 256), strides=(2, 2, 2, 2),
                                                   num_res_units=2, norm=Norm.BATCH)
        self.loss_function = monai.losses.DiceLoss(to_onehot_y=True, softmax=True)
        self.best_val_dice = 0
        self.best_val_loss = 1  # 1 - dice_score
        self.best_val_epoch = 0
        self.dataset_loader = 'standard' # or 'cached'
        self.batch_size = 1

    def forward(self, x):
        return self._model(x)

    def prepare_data(self, dataset = 'custom_head_CT_here'): # 'spleen'):
        # set up the correct data path
        base_dir = '/home/petteri'
        if dataset == 'spleen':
            data_root = os.path.join(base_dir, 'Task09_Spleen')
            train_images = sorted(glob.glob(os.path.join(data_root, 'imagesTr', '*.nii.gz'))) # 512 x 512 x z voxels
            train_labels = sorted(glob.glob(os.path.join(data_root, 'labelsTr', '*.nii.gz')))
            data_dicts = [{'image': image_name, 'label': label_name}
                      for image_name, label_name in zip(train_images, train_labels)]
            train_files, val_files = data_dicts[:-9], data_dicts[-9:]
            test_files = val_files # placeholder
            a_min = - 57
            a_max = 164
            b_min = 0
            b_max = 1
            patch_size = (96,96,96)
        elif dataset == 'custom_head_CT_here':
            data_root = os.path.join(base_dir, 'headCT_dataset')
            train_images = sorted(glob.glob(os.path.join(data_root, 'imagesTr', '*.nii.gz'))) # 256 x 256 x 256 vx
            train_labels = sorted(glob.glob(os.path.join(data_root, 'labelsTr', '*.nii.gz')))
            val_images = sorted(glob.glob(os.path.join(data_root, 'imagesVal', '*.nii.gz')))  # 256 x 256 x 256 vx
            val_labels = sorted(glob.glob(os.path.join(data_root, 'labelsVal', '*.nii.gz')))
            test_images = sorted(glob.glob(os.path.join(data_root, 'imagesTs', '*.nii.gz')))  # 256 x 256 x 256 vx
            test_labels = sorted(glob.glob(os.path.join(data_root, 'labelsTs', '*.nii.gz')))
            train_files = [{'image': image_name, 'label': label_name}
                          for image_name, label_name in zip(train_images, train_labels)] # dict_train
            val_files = [{'image': image_name, 'label': label_name}
                          for image_name, label_name in zip(val_images, val_labels)] # dict_val
            test_files = [{'image': image_name, 'label': label_name}
                          for image_name, label_name in zip(test_images, test_labels)] # dict_test
            # Limits from https://doi.org/10.1016/S2589-7500(20)30085-6
            a_min = - 15
            a_max = 100
            b_min = -1
            b_max = 1
            patch_size = (96, 96, 96)

        print('Dataset "{}": {} train files, {} validation files, {} test files'.format(
            dataset, len(train_files), len(val_files), len(test_files)))

        # set deterministic training for reproducibility
        set_determinism(seed=0)

        # define the data transforms
        train_transforms = Compose([
            LoadNiftid(keys=['image', 'label']),
            Rotated(keys=['image', 'label'], angle=90),
            AddChanneld(keys=['image', 'label']),
            Spacingd(keys=['image', 'label'], pixdim=(1., 1., 1.)),
            Orientationd(keys=['image', 'label'], axcodes='RAS'),
            ScaleIntensityRanged(keys=['image'], a_min=a_min, a_max=a_max,
                                 b_min=b_min, b_max=b_max,
                                 clip=True),
            RandCropByPosNegLabeld(keys=['image', 'label'], label_key='label',
                                   spatial_size=patch_size,
                                   pos=1, neg=1,
                                   num_samples=4, image_key='image', image_threshold=0),
            # https://github.com/Project-MONAI/MONAI/blob/master/examples/notebooks/3d_image_transforms.ipynb
            # https://www.kaggle.com/phoenix9032/seutao-3d-resnet34-model-image-tabular-monai
            Rand3DElasticd(
                keys=['image', 'label'], mode=('bilinear'), prob=1.0,
                sigma_range=(5, 8),
                magnitude_range=(100, 200),
                spatial_size=patch_size,
                translate_range=(50, 50, 2),
                rotate_range=(np.pi / 36, np.pi / 36, np.pi / 36),  # 5 degrees to each direction
                scale_range=(0.15, 0.15, 0.15),
                padding_mode='border'),
            RandGaussianNoised(keys=['image'], prob=0.5, mean=0.0, std=0.5),
            ToTensord(keys=['image', 'label'])
        ])
        val_transforms = Compose([
            LoadNiftid(keys=['image', 'label']),
            Rotated(keys=['image', 'label'], angle=90),  # so that mouth is down
            AddChanneld(keys=['image', 'label']),
            Spacingd(keys=['image', 'label'], pixdim=(1., 1., 1.)),
            Orientationd(keys=['image', 'label'], axcodes='RAS'),
            ScaleIntensityRanged(keys=['image'], a_min=a_min, a_max=a_max,
                                 b_min=b_min, b_max=b_max,
                                 clip=True),
            ToTensord(keys=['image', 'label'])
        ])
        test_transforms = val_transforms

        print('Using dataset_loader type {} (test set always uncached now)'.format(self.dataset_loader))
        if self.dataset_loader == 'cached':
            # we use cached datasets - these are 10x faster than regular datasets
            self.train_ds = monai.data.CacheDataset(
                data=train_files, transform=train_transforms, cache_rate=1.0, num_workers=4
            )
            self.val_ds = monai.data.CacheDataset(
                data=val_files, transform=val_transforms, cache_rate=1.0, num_workers=4
            )
            # self.test_ds = monai.data.CacheDataset(
            #     data=test_files, transform=test_transforms, cache_rate=1.0, num_workers=4
            # ) # not sure if you want to cache this, as in this implementation, you only run this once after training
            self.test_ds = monai.data.Dataset(data=test_files, transform=test_transforms)

        elif self.dataset_loader == 'standard':
            self.train_ds = monai.data.Dataset(data=train_files, transform=train_transforms)
            self.val_ds = monai.data.Dataset(data=val_files, transform=val_transforms)
            self.test_ds = monai.data.Dataset(data=test_files, transform=test_transforms)

    def train_dataloader(self):
        train_loader = DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True,
                                  num_workers=4, collate_fn=list_data_collate)
        return train_loader

    def val_dataloader(self):
        val_loader = DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=4)
        return val_loader

    def test_dataloader(self):
        # https://pytorch-lightning.readthedocs.io/en/latest/lightning-module.html#add-test-loop
        test_loader = DataLoader(self.test_ds, batch_size=self.self.batch_size, num_workers=self.num_workers)
        return test_loader

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self._model.parameters(), 1e-4)
        lr_scheduler = {'scheduler': ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5),
                        'name': 'LR Scheduler'}
        return [optimizer], [lr_scheduler]

    def training_step(self, batch, batch_idx):
        images, labels = batch['image'], batch['label']
        output = self.forward(images)
        loss = self.loss_function(output, labels)
        tensorboard_logs = {'TRAIN/train_loss': loss.item()}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        images, labels = batch['image'], batch['label']
        roi_size = (160, 160, 160)
        sw_batch_size = 4
        outputs = sliding_window_inference(images, roi_size, sw_batch_size, self.forward)
        loss = self.loss_function(outputs, labels)
        value = compute_meandice(y_pred=outputs, y=labels, include_background=False,
                                 to_onehot_y=True, mutually_exclusive=True)
        tensorboard_logs = {'VAL/val_loss': loss}
        return {'val_loss': loss,
                'val_dice': value,
                'log': tensorboard_logs}

    def validation_epoch_end(self, outputs):
        val_dice = 0
        val_loss = 0
        num_items = 0
        for output in outputs:
            val_dice += output['val_dice'].sum().item()
            num_items += len(output['val_dice'])
            val_loss += output['val_loss'].sum().item()
        mean_val_dice = torch.tensor(val_dice / num_items)
        mean_val_loss = torch.tensor(val_loss / num_items)
        tensorboard_logs = {'VAL/val_dice': mean_val_dice,
                            'VAL/mean_val_loss': mean_val_loss}
        # Petteri original tutorial used "mean_val_dice", but it went to zero weirdly at some point
        # while the loss was actually going down? TODO!
        if mean_val_loss < self.best_val_loss:
            self.best_val_loss = mean_val_loss  # update when loss decreases
            self.best_val_dice = mean_val_dice  # do not track dice, but save the dice for best loss
            self.best_val_epoch = self.current_epoch
            print('Validation loss reduced, a new checkpoint _should be saved_ (Petteri)')
        print(
            'current epoch: {} current mean loss: {:.4f} best mean loss: {:.4f} (best dice at that loss {:.4f}) at epoch {}'.format(
                self.current_epoch, mean_val_loss, self.best_val_loss, self.best_val_dice, self.best_val_epoch))
        return {'val_loss': mean_val_loss, 'log': tensorboard_logs}

    # https://pytorch-lightning.readthedocs.io/en/latest/lightning-module.html#add-test-loop
    def test_step(self, batch, batch_idx):
        images, labels = batch['image'], batch['label']
        roi_size = (160, 160, 160)
        sw_batch_size = 4
        outputs = sliding_window_inference(images, roi_size, sw_batch_size, self.forward)
        loss = self.loss_function(outputs, labels)
        value = compute_meandice(y_pred=outputs, y=labels, include_background=False,
                                 to_onehot_y=self.one_hot, mutually_exclusive=True)
        return {'test_loss': loss, 'test_dice': value}

    def test_epoch_end(self, outputs):
        test_dice = 0
        test_loss = 0
        num_items = 0
        for output in outputs:
            test_dice += output['test_dice'].sum().item()
            num_items += len(output['test_dice'])
            test_loss += output['test_loss'].sum().item()
        mean_test_dice = torch.tensor(test_dice / num_items)
        mean_test_loss = torch.tensor(test_loss / num_items)
        return {'mean_val_dice': mean_test_dice, 'mean_test_loss': mean_test_loss}

## Run the training
# initialise the LightningModule
net = Net()

# set up loggers and checkpoints
tb_logger = loggers.TensorBoardLogger(save_dir='logs')
checkpoint_callback = ModelCheckpoint(filepath='logs/{epoch}-{val_loss:.2f}-{val_dice:.2f}',
                                      monitor='val_loss')
lr_logger = LearningRateLogger() # see when your lr was reduced
early_stop_callback = EarlyStopping(
    monitor='val_loss',
    min_delta=0.00,
    patience=20,  # e.g 10 epochs
    verbose=False,
    mode='min'
)

# initialise Lightning's trainer.
trainer = Trainer(gpus=[1],
                  max_epochs=600,
                  early_stop_callback=early_stop_callback,
                  logger=tb_logger,
                  checkpoint_callback=checkpoint_callback,
                  callbacks=[lr_logger],
                  show_progress_bar=False,
                  num_sanity_val_steps=1
                  )
# train
trainer.fit(net)

# Test
# https://pytorch-lightning.readthedocs.io/en/latest/test_set.html#test-after-fit
trainer.test()