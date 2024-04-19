# Copyright 2022 Dakewe Biotech Corporation. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import argparse
from distutils.util import strtobool
import importlib.util
import os
import shutil
import time
from enum import Enum

import torch
import torch.optim as optim
from torch import nn
from torch.cuda import amp
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.text import CharErrorRate, WordErrorRate

from ocrpy.dataset import (
    ICDAR2013Dataset, ICDAR2015Dataset,
    MJSynthDataset, IIIT5KDataset, TRDGDataset,
    train_collate_fn, valid_test_collate_fn,
)
from ocrpy.decoder import ctc_decode
from ocrpy.model import CRNN

_Dataset = {
    "trdg": TRDGDataset,
    "mjsynth": MJSynthDataset,
    "ic15": ICDAR2015Dataset,
    "ic13": ICDAR2013Dataset,
}

def main(config):
    # Initialize the number of training epochs
    start_epoch = 0

    # Initialize training network evaluation indicators
    best_cer, best_wer = 1.0, 1.0

    train_dataloaders, test_dataloader = load_dataset()
    print("Load all datasets successfully.")

    model = build_model()
    from torchinfo import summary; summary(model)
    print("Build CRNN model successfully.")

    criterion = define_loss()
    print("Define all loss functions successfully.")

    optimizer, scheduler = define_optimizer(model)
    print("Define all optimizer functions successfully.")

    cer = CharErrorRate()
    wer = WordErrorRate()
    print("Define all metrics successfully.")

    print("Check whether the pretrained model is restored...")
    if config.resume:
        # Load checkpoint model
        checkpoint = torch.load(config.resume, map_location=lambda storage, loc: storage)
        # Restore the parameters in the training node to this point
        start_epoch = checkpoint["epoch"]
        best_cer = checkpoint["best_cer"]
        best_wer = checkpoint["best_wer"]
        # Load checkpoint state dict. Extract the fitted model weights
        model_state_dict = model.state_dict()
        new_state_dict = {k: v for k, v in checkpoint["state_dict"].items() if k in model_state_dict.keys()}
        # Overwrite the pretrained model weights to the current model
        model_state_dict.update(new_state_dict)
        model.load_state_dict(model_state_dict)
        # Load the optimizer model
        optimizer.load_state_dict(checkpoint["optimizer"])
        print("Loaded pretrained model weights.")

    # Create a folder of super-resolution experiment results
    samples_dir = os.path.join("samples", config.exp_name)
    results_dir = os.path.join("results", config.exp_name)
    if not os.path.exists(samples_dir):
        os.makedirs(samples_dir)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Create training process log file
    writer = SummaryWriter(os.path.join("logs", config.exp_name))

    # Initialize the gradient scaler
    scaler = amp.GradScaler()

    for epoch in range(start_epoch, config.epochs):
        train(
            model, train_dataloaders, test_dataloader,
            cer, wer, criterion,
            epoch,
            optimizer, scheduler, scaler, writer
        )
        cur_cer, cur_wer = validate(
            model, test_dataloader,
            cer, wer, epoch, writer, "test"
        )
        print(f"Skipped corrupted data: {train_dataloaders[0].dataset.num_skipped}")

        #if not epoch or epoch % 5 != 0: continue
        is_best_cer, is_best_wer = cur_cer < best_cer, cur_wer < best_wer
        best_cer, best_wer = min(cur_cer, best_cer), min(cur_wer, best_wer)

        torch.save({"epoch": epoch + 1,
                    "best_wer": best_wer,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict()},
                   os.path.join(samples_dir, f"epoch_{epoch + 1}.pth.tar"))
        if is_best_cer:
            shutil.copyfile(os.path.join(samples_dir, f"epoch_{epoch + 1}.pth.tar"),
                            os.path.join(results_dir, "best_cer.pth.tar"))
        if is_best_wer:
            shutil.copyfile(os.path.join(samples_dir, f"epoch_{epoch + 1}.pth.tar"),
                            os.path.join(results_dir, "best_wer.pth.tar"))
        if (epoch + 1) == config.epochs:
            shutil.copyfile(os.path.join(samples_dir, f"epoch_{epoch + 1}.pth.tar"),
                            os.path.join(results_dir, "last.pth.tar"))


def load_dataset() -> [DataLoader, DataLoader]:
    TrainDataset = MJSynthDataset
    TestDataset = ICDAR2013Dataset
    # Load train and test datasets
    train_datasets = []
    for ds_root, ds_type in zip(config.train_dataroot, config.train_dataset):
        train_datasets.append(
            _Dataset[ds_type](
                dataroot=ds_root,
                labels_dict=config.labels_dict,
                image_width=config.model_image_width,
                image_height=config.model_image_height,
                mean=config.mean,
                std=config.std,
                mode="train"
            )
        )
    test_dataset = _Dataset["ic13"](
        dataroot=config.test_dataroot,
        image_width=config.model_image_width,
        image_height=config.model_image_height,
        mean=config.mean,
        std=config.std,
        mode="test"
    )
 
    # Generator all dataloader
    train_dataloaders = []
    for ds, bsz in zip(train_datasets, config.batch_size):
        train_dataloaders.append(
            DataLoader(
                dataset=ds,
                batch_size=bsz, # mix 2 datasets
                shuffle=True,
                num_workers=config.num_workers,
                collate_fn=train_collate_fn,
                pin_memory=True,
                drop_last=True,
                persistent_workers=True
            )
        )

    test_dataloader = DataLoader(dataset=test_dataset,
                                 batch_size=sum(config.batch_size),
                                 shuffle=False,
                                 num_workers=1,
                                 collate_fn=valid_test_collate_fn,
                                 pin_memory=True,
                                 drop_last=False,
                                 persistent_workers=True)

    return train_dataloaders, test_dataloader


def build_model() -> nn.Module:
    model = CRNN(config.model_num_classes)
    model = model.to(device=config.device)

    return model


def define_loss() -> nn.CTCLoss:
    criterion = nn.CTCLoss(zero_infinity=True)
    criterion = criterion.to(device=config.device)

    return criterion


def define_optimizer(model) -> optim.Adadelta:
    #optimizer = optim.Adadelta(model.parameters(), config.model_lr)
    #optimizer = optim.RMSprop(model.parameters(), config.model_lr, weight_decay=0.01)
    optimizer = optim.Adam(model.parameters(), config.model_lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, config.epochs * config.batches_per_epoch, eta_min=1e-6)

    return optimizer, scheduler


def train(model: nn.Module,
          train_dataloaders: list,
          test_dataloader: DataLoader,
          cer: CharErrorRate,
          wer: WordErrorRate,
          criterion: nn.CTCLoss,
          epoch: int,
          optimizer: optim.RMSprop,
          scheduler: optim.lr_scheduler.CosineAnnealingLR,
          scaler: amp.GradScaler,
          writer: SummaryWriter) -> None:
    """Training main program

    Args:
        model (nn.Module): CRNN model
        train_dataloader (list[DataLoader]): training dataset iterator
        criterion (nn.CTCLoss): Calculates loss between a continuous (unsegmented) time series and a target sequence
        epoch (int): number of training epochs during training the generative network
        optimizer (optim.RMSprop): optimizer for optimizing generator models in generative networks
        scheduler (optim.lr_scheduler.CosineAnnealingLR)
        scaler (amp.GradScaler): Mixed precision training function
        writer (SummaryWrite): log file management function

    """
    # Calculate how many batches of data are in each Epoch
    batches = min([len(tdl) for tdl in train_dataloaders])
    # Print information of progress bar during training
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":6.6f")
    lrs = AverageMeter("LR", ":1.3e")
    progress = ProgressMeter(batches, [batch_time, data_time, losses, lrs], prefix=f"Epoch: [{epoch + 1}]")


    # Get the initialization training time
    end = time.time()

    for batch_index, batch in enumerate(zip(*train_dataloaders)):
        if len(train_dataloaders) == 1:
           images, targets, target_lengths = batch[0]
        else: raise NotImplementedError
        # Put the generative network model in training mode
        model.train()

        # Calculate the time it takes to load a batch of data
        data_time.update(time.time() - end)

        # Get the number of data in the current batch
        curren_batch_size = images.size(0)

        # Transfer in-memory data to CUDA devices to speed up training
        images = images.to(device=config.device, non_blocking=True)
        targets = targets.to(device=config.device, non_blocking=True)
        target_lengths = target_lengths.to(device=config.device, non_blocking=True)

        # Initialize generator gradients
        model.zero_grad(set_to_none=True)

        # Mixed precision training
        if config.amp:
            with amp.autocast():
                outputs = model(images)
                output_log_probs = F.log_softmax(outputs, 2)
                image_lengths = torch.tensor(
                    [outputs.size(0)] * curren_batch_size,
                     dtype=torch.long, device=output_log_probs.device
                )
                #target_lengths = torch.flatten(target_lengths)
                # Computational loss
                loss = criterion(output_log_probs, targets, image_lengths, target_lengths)
        else:
            outputs = model(images)
            output_log_probs = F.log_softmax(outputs, 2)
            image_lengths = torch.tensor(
                [outputs.size(0)] * curren_batch_size,
                 dtype=torch.long, device=output_log_probs.device
            )
            #target_lengths = torch.flatten(target_lengths)
            # Computational loss
            loss = criterion(output_log_probs, targets, image_lengths, target_lengths)

        # Backpropagation
        if config.amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
        else:
            loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        # update generator weights
        if config.amp:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        # Statistical loss value for terminal data output
        losses.update(loss.item(), curren_batch_size)

        # Calculate the time it takes to fully train a batch of data
        batch_time.update(time.time() - end)
        end = time.time()

        scheduler.step()
        lrs.update(scheduler.get_last_lr()[0], curren_batch_size)

        # Write the data during training to the training log file
        if batch_index % config.print_frequency == 0:
            # Record loss during training and output to file
            writer.add_scalar("Train/Loss", loss.item(), batch_index + epoch * batches + 1)
            writer.add_scalar("Train/LR", scheduler.get_last_lr()[0], batch_index + epoch * batches + 1)
            progress.display(batch_index)
            _ = validate(model, test_dataloader, cer, wer, batch_index + epoch * batches, writer, "valid")


def validate(
        model: nn.Module,
        dataloader: DataLoader,
        cer: CharErrorRate,
        wer: WordErrorRate,
        epoch: int,
        writer: SummaryWriter,
        mode: str
    ) -> [float, float]:
    """Test main program

    Args:
        model (nn.Module): CRNN model
        dataloader (DataLoader): Test dataset iterator
        epoch (int): Number of test epochs during training of the adversarial network
        writer (SummaryWriter): Log file management function
        mode (str): test validation dataset accuracy or test dataset accuracy
    """
    # Put the adversarial network model in validation mode
    model.eval()

    with torch.no_grad():
        for image_paths, images, labels in dataloader:
            # Transfer in-memory data to CUDA devices to speed up training
            images = images.to(device=config.device, non_blocking=True)

            # Mixed precision testing
            with amp.autocast():
                output = model(images)
            output_log_probs = F.log_softmax(output, 2)
            _, prediction_chars = ctc_decode(output_log_probs, config.chars_dict)

            # Record CER
            # TODO: implement case sensitivity
            for b in range(len(images)):
                cur_cer = cer("".join(prediction_chars[b]), labels[b].lower())
                cur_wer = wer("".join(prediction_chars[b]), labels[b].lower())
                if cur_cer > 0.0 and mode == "test":
                    information = (
                        f"`{image_paths[b].name}`: "
                        f"`{labels[b].lower()}` -> "
                        f"`{''.join(prediction_chars[b])}`"
                    )
                    print(information)

    total_cer = cer.compute()
    total_wer = wer.compute()
    information = f"CER: {total_cer *100: .2f}%"
    information += f"\tWER: {total_wer *100: .2f}%"
    print(information)
    cer.reset()
    wer.reset()

    if mode == "valid" or mode == "test":
        writer.add_scalar(f"{mode}/CER", total_cer, epoch+1)
        writer.add_scalar(f"{mode}/WER", total_wer, epoch+1)
    else:
        raise ValueError("Unsupported mode, please use `valid` or `test`.")
    return total_cer, total_wer


class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3


class AverageMeter(object):
    def __init__(self, name, fmt=":f", summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)

    def summary(self):
        if self.summary_type is Summary.NONE:
            fmtstr = ""
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = "{name} {avg:.2f}"
        elif self.summary_type is Summary.SUM:
            fmtstr = "{name} {sum:.2f}"
        elif self.summary_type is Summary.COUNT:
            fmtstr = "{name} {count:.2f}"
        else:
            raise ValueError(f"Invalid summary type {self.summary_type}")

        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))

    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(" ".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--name", type=str, help="Experiment name")
    # Data
    parser.add_argument("--batch", type=int, default=512, help="Batch size")
    # Training
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--epoch", type=int, default=5, help="Number of epochs")
    parser.add_argument("--amp", dest="amp", type=lambda x: bool(strtobool(x)), default=True, help="AMP training")


    args = parser.parse_args()
    # Load config python script
    config_spec = importlib.util.spec_from_file_location("config", args.config)
    config = importlib.util.module_from_spec(config_spec)
    config_spec.loader.exec_module(config)
    config.exp_name = args.name
    config.batch_size = [args.batch]
    config.batches_per_epoch = config.samples / sum(config.batch_size) 
    config.model_lr = args.lr
    config.epochs = args.epoch
    config.amp = args.amp
    return config

if __name__ == "__main__":
    config = get_args() 
    main(config)
