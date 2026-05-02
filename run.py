import os
import sys
import yaml
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import lightning as L
from lightning import Trainer
from lightning.pytorch.callbacks import RichProgressBar, ModelCheckpoint
from lightning.pytorch.callbacks.progress.rich_progress import RichProgressBarTheme
from lightning.pytorch.loggers import TensorBoardLogger

from data.scene_dataset import DataModule
from parse_args import Args, get_parser
from control_tile import ControlTileModel
from utils import RepeatingTimer

def train(args: Args):
    is_debug = 'debug' in args.stage
    data_module = DataModule(args)
    data_module.setup()

    model = ControlTileModel(args)

    # RichProgressBar is not working with slurm
    logger = TensorBoardLogger(args.logging_dir, name="train", version=args.output_name)
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=args.model_output,
        every_n_epochs=args.val_per_epoch,
        save_last=True,
        save_top_k=args.checkpoints_total_limit,
        filename=args.output_name + '-{epoch}-{step}'
    )
    trainer = Trainer(
        logger = logger,
        callbacks=[checkpoint_callback],
        max_epochs=args.num_train_epochs,
        check_val_every_n_epoch=args.val_per_epoch,
        precision='bf16-mixed',
        devices=args.devices,
        accumulate_grad_batches=8,
        strategy='ddp',
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
        enable_model_summary=True,
        enable_checkpointing=True,
        fast_dev_run=is_debug,
    )

    trainer.fit(model, data_module, ckpt_path=args.resume)

def test(args: Args):
    data_module = DataModule(args)
    data_module.setup()
    test_dataloader = data_module.test_dataloader()
    model = TpdmModel(args)
    state_dict = torch.load(args.ckpt, weights_only=True)['state_dict']
    model.load_state_dict(state_dict, strict=False)

    if args.seed is not None:
        L.seed_everything(args.seed)
    
    progress_bar = RichProgressBar(
        theme=RichProgressBarTheme(
            description="green_yellow",
            progress_bar="green1",
            progress_bar_finished="green1",
            progress_bar_pulse="#6206E0",
            batch_progress="green_yellow",
            time="grey82",
            processing_speed="grey82",
            metrics="grey82",
            metrics_text_delimiter=" | ",
            metrics_format=".3e",
        )
    )

    logger = TensorBoardLogger(args.logging_dir, name="test", version='test')

    trainer = Trainer(
        logger=logger,
        precision='bf16-mixed',
        callbacks=[progress_bar],
        # enable_progress_bar=False,
        devices=args.devices,
    )

    os.makedirs(args.img_output_folder, exist_ok=True)
    trainer.test(model, dataloaders=test_dataloader)

if __name__ == '__main__':

    print('[cuda available]: ' + str(torch.cuda.is_available()))
    print('[cuda version]: ' + torch.version.cuda)
    print('[cudnn version]: ' + str(torch.backends.cudnn.version()))

    import sys

    config = sys.argv[1]

    with open('config/config.yaml', 'r') as f:
        default_arg: dict = yaml.safe_load(f)
    parser = get_parser()
    parser.set_defaults(**default_arg.get(config))
    args = Args(parser.parse_args(sys.argv[2:]))

    try:
        if 'train' in config or 'debug' in config:
            train(args)
        elif 'test' in config:
            test(args)
    except Exception as e:
        print(e)
        raise e
