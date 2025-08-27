import os
from collections import defaultdict
from datetime import datetime

import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateFinder
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from datamodules.wsi_embedding_datamodule import PatchEmbeddingDataModule, EmbeddingDataModule
from pytorch_lightning.tuner.tuning import Tuner

from sklearn.model_selection import KFold

from models import ReportModel
import torch

class Trainer:

    def __init__(self, args, tokenizer, split_frac):
        self.best_model_path = None
        self.best=0
        self.ckpt_path = args.ckpt_path
        self.max_epochs = args.max_epochs
        self.split_frac = split_frac
        self.datamodule = PatchEmbeddingDataModule(args, tokenizer, split_frac)
        # self.model = ReportModel(args, tokenizer)
        pl.seed_everything(42)
        # torch.set_float32_matmul_precision('high')
        # torch.use_deterministic_algorithms(True)
        self.trainer = None
        self.devices = args.devices if type(args.devices)==int else list(map(int, args.devices.split(',')))
        self.args = args
        self.tokenizer = tokenizer

    def train(self, model, datamodule, fast_dev_run=False):
        # datamodule = PatchEmbeddingDataModule(self.args, self.tokenizer, self.split_frac)
        # ts = datetime.now().strftime("%Y%m%d")
        # ckpt_path = f'{self.ckpt_path}/{ts}'

        checkpoint_callback = ModelCheckpoint(
            dirpath=self.args.ckpt_path,  # Directory to save checkpoints
            filename="model_{epoch:02d}_{val_loss:.5f}_{val_reg:.5f}",  # Naming convention
            monitor="val_loss",  # Metric to monitor for saving best checkpoints
            mode="min",  # Whether to minimize or maximize the monitored metric
            save_top_k=1,  # Number of best checkpoints to keep
            save_last=True  # Save the last checkpoint regardless of the monitored metric
        )
        early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-5, patience=7, verbose=True, mode="min")

        # lr_finder = LearningRateFinder(
        #     min_lr=1e-8,  # Minimum learning rate to test
        #     max_lr=1e-4,  # Maximum learning rate to test
        #     num_training_steps=100,  # Number of learning rates to test
        #     mode='exponential'  # or 'linear'
        # )

        suggested_lr = self.find_lr(model, datamodule)
        print(f'setting lr: {suggested_lr}')
        # Set suggested LR
        model.hparams.lr = suggested_lr

        self.trainer = pl.Trainer(
            max_epochs=self.max_epochs,
            callbacks=[checkpoint_callback, early_stop_callback],
            accelerator='gpu',
            devices=self.devices,
            strategy='ddp_find_unused_parameters_true',
            enable_progress_bar=True,
            log_every_n_steps=1,
            fast_dev_run=fast_dev_run
        )

        self.trainer.fit(
            model, datamodule=datamodule
        )

        train_metrics = self.trainer.logged_metrics
        self.best_model_path = checkpoint_callback.best_model_path
        self.best = train_metrics['val_loss']
        return train_metrics, self.trainer


    def find_lr(self, model, datamodule):
        trainer = pl.Trainer(
            accelerator='gpu',
            devices=self.devices,
            strategy='ddp_find_unused_parameters_true',
            enable_progress_bar=True
        )
        tuner = Tuner(trainer)
        lr_finder = tuner.lr_find(model, datamodule=datamodule)

        suggested_lr = lr_finder.suggestion()
        print(f'suggested lr: {suggested_lr}')

        return suggested_lr

    def tune(self, model, datamodule, fast_dev_run=False):
        # datamodule = PatchEmbeddingDataModule(self.args, self.tokenizer, self.split_frac)
        # ts = datetime.now().strftime("%Y%m%d")
        # ckpt_path = f'{self.ckpt_path}/{ts}'

        model.model.freeze_deep_features()

        checkpoint_callback = ModelCheckpoint(
            dirpath=self.args.ckpt_path,  # Directory to save checkpoints
            filename="model_tune_{epoch:02d}_{val_loss:.5f}_{val_reg:.5f}",  # Naming convention
            monitor="val_loss",  # Metric to monitor for saving best checkpoints
            mode="min",  # Whether to minimize or maximize the monitored metric
            save_top_k=1,  # Number of best checkpoints to keep
            save_last=True  # Save the last checkpoint regardless of the monitored metric
        )
        early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-5, patience=5, verbose=True, mode="min")
        # lr_finder = LearningRateFinder(
        #     min_lr=1e-8,  # Minimum learning rate to test
        #     max_lr=1e-5,  # Maximum learning rate to test
        #     num_training_steps=50,  # Number of learning rates to test
        #     mode='exponential'  # or 'linear'
        # )
        suggested_lr = self.find_lr(model, datamodule)
        print(f'setting lr: {suggested_lr}')
        # Set suggested LR
        model.hparams.lr = suggested_lr
        self.trainer = pl.Trainer(
            max_epochs=30,
            callbacks=[checkpoint_callback, early_stop_callback],
            accelerator='gpu',
            devices=self.devices,
            strategy='ddp_find_unused_parameters_true',
            enable_progress_bar=True,
            log_every_n_steps=1,
            fast_dev_run=fast_dev_run
        )

        self.trainer.fit(
            model, datamodule=datamodule
        )

        tune_metrics = self.trainer.logged_metrics

        self.best_model_path = checkpoint_callback.best_model_path
        return tune_metrics, self.trainer

    def test(self, model, datamodule, fast_dev_run=False):

        trainer = pl.Trainer(
            accelerator='gpu',
            devices=self.devices,
            strategy='ddp_find_unused_parameters_true',
            enable_progress_bar=True,
            log_every_n_steps=1,
            fast_dev_run=fast_dev_run
        )

        trainer.test(
            model, datamodule=datamodule
        )
        test_metrics = trainer.logged_metrics
        return test_metrics, trainer

    def predict(self, model, datamodule, fast_dev_run=False):

        trainer = pl.Trainer(
            accelerator='gpu',
            devices=self.devices,
            strategy='ddp_find_unused_parameters_true',
            enable_progress_bar=True,
            log_every_n_steps=1,
            fast_dev_run=fast_dev_run
        )

        preds = trainer.predict(
            model, datamodule=datamodule
        )
        # print(f'leng preds: {len(preds)}')
        # flat_preds = [p for sublist in preds for p in sublist]
        return preds

    @rank_zero_only
    def save_model(self,trainer, model_path):

        trainer.save_checkpoint(model_path)
        print(f'model saved at path: {model_path}')

    def load_model(self, model_cls, model_path, **kwargs):
        return model_cls.load_from_checkpoint(model_path, **kwargs)


class KFoldTrainer(Trainer):
    def __init__(self, args, tokenizer, split_frac):
        super().__init__(args, tokenizer, split_frac)

        self.__kf = KFold(n_splits=args.num_folds, shuffle=True, random_state=42)

        self.split_frac =split_frac
        self.best_models = []
        self.train_metrics=defaultdict(list)
        self.test_metrics=defaultdict(list)

    def get_metrics(self):
        print(f'train_metrics: {self.train_metrics}, test_metrics: {self.test_metrics}')

        train_metrics = {
            metric: f'{np.mean(value)} \u00B1 {np.std(value)}' for metric, value in self.train_metrics.items()
        }

        test_metrics = {
            metric: f'{np.mean(value)} \u00B1 {np.std(value)}' for metric, value in self.test_metrics.items()
        }

        metrics = {**train_metrics, **test_metrics, 'best_model_path': self.get_best_model_path()}
        return metrics

    def get_best_model_path(self):
        self.best_models.sort(reverse=True, key=lambda x: x[1])
        return self.best_models[0]

    def load_model(self, model_cls, **kwargs):
        model_path = self.get_best_model_path()
        super().load_model(model_cls, model_path, **kwargs)

    @rank_zero_only
    def save_model(self, trainer, model_path):

        trainer.save_checkpoint(model_path)
        print(f'model saved at path: {model_path}')


    def train_and_test(self, fast_dev_run=False):
        files = os.listdir(self.args.embeddings_path)

        for fold, (train_idx, test_idx) in enumerate(self.__kf.split(files)):
            # print(f'__reports: {len(self.__reports)}, train_idx: {len(train_idx)}: {train_idx}, test_idx: {len(test_idx)}: {test_idx}')
            print("*"*100)
            print(f'training for fold: {fold}')
            self.datamodule = EmbeddingDataModule(self.args, self.tokenizer, self.split_frac, train_idx, test_idx)
            ts = datetime.now().strftime("%Y%m%d")
            ckpt_path = f'{self.ckpt_path}/{ts}'
            checkpoint_callback = ModelCheckpoint(
                dirpath=ckpt_path,  # Directory to save checkpoints
                filename=f"fold{fold}_" + "{epoch:02d}_{val_loss:.5f}",  # Naming convention
                monitor="val_loss",  # Metric to monitor for saving best checkpoints
                mode="min",  # Whether to minimize or maximize the monitored metric
                save_top_k=1,  # Number of best checkpoints to keep
                save_last=True  # Save the last checkpoint regardless of the monitored metric
            )
            early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=5, verbose=True,
                                                mode="min")
            trainer = pl.Trainer(
                max_epochs=self.max_epochs,
                callbacks=[checkpoint_callback, early_stop_callback],
                accelerator='gpu',
                devices=self.devices,
                strategy='ddp_find_unused_parameters_true',
                enable_progress_bar=True,
                log_every_n_steps=2,
                fast_dev_run=fast_dev_run
            )

            model = ReportModel(self.args, self.tokenizer)
            trainer.fit(
                model, datamodule=self.datamodule
            )
            train_metrics = trainer.logged_metrics
            # self.train_metrics[f'fold_{fold}'] = train_metrics
            print(f'fold: {fold}, train_metrics: {train_metrics}')

            for metric,value in  train_metrics.items():
                self.train_metrics[metric].append(value)

            trainer.test(
                model, datamodule=self.datamodule
            )
            test_metrics = trainer.logged_metrics
            # self.test_metrics[f'fold_{fold}'] = test_metrics

            best_model_path = checkpoint_callback.best_model_path
            self.best_models.append((best_model_path, test_metrics['test_reg'], fold))

            print(f'fold: {fold}, test_metrics: {test_metrics}')

            for metric,value in  test_metrics.items():
                self.test_metrics[metric].append(value)

            print(f'Finished!')
            print("*"*100)





