"""Training for the Data-driven Non-linear Smoother (DNS)."""
import argparse
import functools
import logging
import math
import os
import shutil

import gin
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim
import tqdm

import dataset as dataset_lib
import model as model_lib
import utils


@gin.configurable
class TrainingLoop:
    """Training loop for DANSE Markov model."""

    def __init__(
        self,
        danse_state_posterior_model: model_lib.DatadrivenNonlinearSmoother,
        dataset_cls: dataset_lib.BaseDynamicalDataset,
        save_path: str,
        num_epochs: int,
        learning_rate: float,
        weight_decay: float,
        batch_size: int,
        device: str,
        valid_signal_length: int = 100,
    ) -> None:
        """Initialize the training loop."""
        self._save_path = save_path
        self._log_path = os.path.join(save_path, 'train.log')
        logging.basicConfig(filename=self._log_path, level=logging.INFO)
        self._device = torch.device('cuda' if torch.cuda.is_available() and device == 'cuda' else 'cpu')
        logging.info(f'Device: {self._device}')

        self._epoch = 0
        self._num_epochs = num_epochs
        self._learning_rate = learning_rate
        self._batch_size = batch_size

        self._train_dataset = dataset_cls()
        self._train_dataloader = dataset_lib.get_dataloader(
            self._train_dataset,
            self._batch_size,
        )
        self._valid_dataset = dataset_cls(
            num_samples=100,
            signal_length=valid_signal_length
        )
        self._valid_dataloader = dataset_lib.get_dataloader(
            self._valid_dataset,
            self._batch_size,
        )
        self._test_dataset = dataset_cls(
            num_samples=100,
            signal_length=valid_signal_length
        )
        self._test_dataloader = dataset_lib.get_dataloader(
            self._test_dataset,
            self._batch_size,
        )
        self._H = self._train_dataset.H.to(device=self._device)

        self._danse_state_posterior_model = danse_state_posterior_model(
            device=self._device,
        ).to(device=self._device)
        self._optimizer = torch.optim.Adam(
            self._danse_state_posterior_model.parameters(),
            lr=learning_rate,
            eps=1e-8,
            weight_decay=weight_decay,
        )
        self._scheduler = torch.optim.lr_scheduler.StepLR(
            self._optimizer,
            step_size=num_epochs // 6,
            gamma=0.9
        )
        self._nmse_loss_accumulator = utils.NMSELossAccumulator()
        self._loglikelihood_accumulator = utils.LogLikelihood()

        self._state_to_observation = functools.partial(
            self._train_dataset.state_to_observation,
            device=self._device
        )
        self._convergence_monitor = utils.ConvergenceMonitor(
            patience=1000, min_delta=1e-2
        )

        self._best_loss = float('inf')

    def _plot_time_series(
        self,
        targets,
        predictions,
        variable_name,
        prefix,
        noisy=None
    ):
        plt.plot(targets, label='True')
        if noisy is not None:
            plt.plot(noisy, label='Noisy')
        plt.plot(predictions, label='Predicted')
        plt.title(f'{variable_name} curve over timesteps')
        plt.xlabel('Timesteps')
        plt.ylabel(f'{variable_name}')
        plt.legend()
        plt.savefig(os.path.join(
            self._save_path, 
            f'{prefix}_{variable_name}_epoch={self._epoch}.png'
        ))
        plt.close()

    def _calculate_loss(
        self,
        mu_prior: torch.Tensor,
        cov_prior: torch.Tensor,
        noisy_observations: torch.Tensor,
        Cws: torch.Tensor,
    ) -> torch.Tensor:
        H = self._H

        posterior_cov = Cws + H @ cov_prior @ H.transpose(-1, -2)
        error = noisy_observations - torch.einsum('ij,ntj->nti', H, mu_prior)
        weighted_mse_loss = torch.einsum(
            'nti,nti->nt',
            error,
            torch.einsum('ntij,ntj->nti', torch.inverse(posterior_cov), error)
        )
        log_likelihood = weighted_mse_loss + torch.logdet(posterior_cov)
        return torch.mean(log_likelihood)

    def predict_state_markov_model(
        self,
        noisy_observations: torch.Tensor,
        Cws: torch.Tensor,
    ) -> torch.Tensor:

        _, _, mu_posterior, _ = self._danse_state_posterior_model(
            noisy_observations,
            Cws,
        )
        return mu_posterior

    def train(self):
        while self._epoch < self._num_epochs:
            loss_epoch = 0.0
            self._danse_state_posterior_model.train()
            for batch in tqdm.tqdm(
                self._train_dataloader,
                total=len(self._train_dataloader),
                ncols=0,
                desc="Train"
            ):
                _, _, noisy_observations, Cws = batch
                noisy_observations = noisy_observations.to(device=self._device)
                Cws = Cws.to(device=self._device)
                mu_prior, cov_prior, *_  = self._danse_state_posterior_model(
                    noisy_observations,
                    Cws
                )
                loss = self._calculate_loss(
                    mu_prior,
                    cov_prior,
                    noisy_observations,
                    Cws,
                )

                loss_epoch += loss.item()
                loss.backward()
                self._optimizer.step()
                self._optimizer.zero_grad()
            self._epoch += 1
            self._scheduler.step()

            avg_loss = loss_epoch / len(self._train_dataloader)
            logging.info(
                f'Epoch={self._epoch}, Average loss train={avg_loss:.4f}, '
            )
            self.valid()
            if self._convergence_monitor.update(avg_loss):
                logging.info(
                    f'Early stopping at epoch {self._epoch} due to convergence.'
                )
                self._epoch = self._num_epochs  # Stop training

    def _plot_validation_trajectory_per_dimension(
        self,
        dataloader,
        prefix,
    ) -> None:
        _, true_observations, noisy_observations, Cws = next(iter(dataloader))
        noisy_observations = noisy_observations.to(device=self._device)
        Cws = Cws.to(device=self._device)
        predicted_states_markov = self.predict_state_markov_model(
            noisy_observations,
            Cws,
        )
        noisy_observations = noisy_observations.detach().cpu()
        predicted_states_markov = predicted_states_markov.detach().cpu()

        *_, dimensions = true_observations.shape
        for dimension in range(dimensions):
            x_true = true_observations[0, :, dimension]
            x_noisy = noisy_observations[0, :, dimension]
            x_pred_markov = predicted_states_markov[0, :, dimension]
            self._plot_time_series(
                x_true,
                x_pred_markov,
                f'x_{dimension}',
                noisy=x_noisy,
                prefix=prefix
            )

    def _evaluate(
        self,
        dataloader,
        prefix: str,
        save_if_best: bool = True,
    ) -> None:
        self._danse_state_posterior_model.eval()
        total_loss = 0
        self._nmse_loss_accumulator.reset()
        self._loglikelihood_accumulator.reset()
        for batch in tqdm.tqdm(
            dataloader,
            total=len(dataloader),
            ncols=0,
            desc=prefix
        ):
            states, _, noisy_observations, Cws = batch
            noisy_observations = noisy_observations.to(device=self._device)
            Cws = Cws.to(device=self._device)
            mu_prior, cov_prior, mu_posterior, cov_posterior = self._danse_state_posterior_model(
                noisy_observations,
                Cws,
            )
            loss = self._calculate_loss(
                mu_prior,
                cov_prior,
                noisy_observations,
                Cws,
            )
            total_loss += loss.item()
            
            with torch.no_grad():
                predictions = self.predict_state_markov_model(
                    noisy_observations,
                    Cws,
                ).cpu()
            self._nmse_loss_accumulator.update(predictions, states)
            self._loglikelihood_accumulator.update(
                mu_posterior.cpu(),
                cov_posterior.cpu(),
                states
            )
        nmse_loss = self._nmse_loss_accumulator.get_total_nmse_loss()
        logging.info(
            f'[{prefix}] NMSE = {nmse_loss:.3f} dB'
        )
        log_likelihood = self._loglikelihood_accumulator.get_total_log_likelihood()
        logging.info(
            f'[{prefix}] Log likelihood = {log_likelihood:.3f}'
        )
        avg_total_loss = total_loss / len(dataloader)
        avg_loss = loss.item() / len(dataloader)
        logging.info(
            f'[{prefix}] Avg loss={avg_loss:.4f}'
        )
        if self._epoch % 50 == 0:
            self._plot_validation_trajectory_per_dimension(
                dataloader,
                prefix,
            )
        if save_if_best and avg_total_loss < self._best_loss:
            self._best_loss = avg_total_loss
            logging.info(
                f'New best loss: {self._best_loss:.4f}, saving model...'
            )
            self.save_model(suffix='best')
    
    def valid(self) -> None:
        """Validate the model on the validation dataset."""
        logging.info(f'Epoch {self._epoch}: Validating...')
        self._evaluate(
            self._valid_dataloader,
            prefix='Valid',
            save_if_best=True
        )
    
    def test(self) -> None:
        """Test the model on the test dataset."""
        logging.info(f'Epoch {self._epoch}: Testing...')
        state_dict = torch.load(
            os.path.join(self._save_path, 'danse_model_state_dict_best.pt'),
            map_location=self._device
        )
        self._danse_state_posterior_model.load_state_dict(state_dict)
        self._evaluate(
            self._test_dataloader,
            prefix='Test',
            save_if_best=False,
        )

    def save_model(self, suffix: str = 'last') -> None:
        torch.save(
            self._danse_state_posterior_model.state_dict(),
            os.path.join(self._save_path, f'danse_model_state_dict_{suffix}.pt')
        )

                
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_path', type=str, required=True)
    parser.add_argument('--gin_path', type=str, required=True)
    args = parser.parse_args()

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    gin.parse_config_file(args.gin_path)
    gin.finalize()
    shutil.copy(args.gin_path, os.path.join(args.save_path, 'gin_config.gin'))
    training_loop = TrainingLoop(save_path=args.save_path)
    training_loop.train()
    training_loop.test()


if __name__ == '__main__':
    main()
