"""Dataset definition and loader.

To make it easier to implement datasets into the framework, there is a baseclass
for datasets, called `BaseDynamicalDataset`. This class implements the
`torch.utils.data.Dataset` interface. Datasets suitable for the framework have
the following:
* A temporal component, and the states and observations are 1D tensors.
    The latter means that images, videos, etc. are not suitable. A trajectory
    (state/observation) is of shape (T, D), where T is the number of time steps
    and D is the dimension.
* A linear state to observation mapping.

Minimal example of implementing a dataset:

```
class NewDataset(BaseDynamicalDataset):

    def __init__(self):
        super().__init__(
            num_samples=100,
            signal_length=10,
            smnr_db=10,
            H=torch.eye(2)
        )

    @property
    def observation_noise_covariance(self):
        return torch.eye(2) * 0.1

    def _generate_data(self):
        states = torch.ones(self._num_samples, self._signal_length, 2)
        observations = self.state_to_observation(states)
        mvn = torch.distributions.multivariate_normal.MultivariateNormal(
            torch.zeros(2),
            self.observation_noise_covariance,
        )
        noisy_observations = observations + mvn.sample(
            (self._num_samples, self._signal_length)
        )
        return states, observations, noisy_observations
```
"""
import abc
from typing import Sequence

import gin
import numpy as np
import pandas as pd
import torch
import torch.utils.data
import torch.utils.data.dataset

import double_pendulum
import lorenz_attractor


def _check_same_shape_and_expected_dim(
    *arrays: np.ndarray | torch.Tensor,
    expected_dim: int
) -> None:
    """Checks that all arrays have the same shape and `expected_dim` dimensions.

    Args:
        arrays: The arrays to check.
        expected_dim: The expected number of dimensions.

    Raises:
        ValueError: If the arrays do not have the same shape or number of dimensions.
    """
    first_shape = arrays[0].shape
    for array in arrays:
        if array.shape != first_shape:
            raise ValueError(
                f"Arrays have different shapes: {first_shape} vs {array.shape}"
            )
        if len(array.shape) != expected_dim:
            raise ValueError(
                f"Array has {len(array.shape)} dimensions, expected {expected_dim}"
            )

def _convert_arrays_to_tensors(*arrays: np.ndarray) -> list[torch.Tensor]:
    """Converts numpy arrays to PyTorch tensors.
    
    Args:
        arrays: The arrays to convert.
    
    Returns:
        The arrays converted to PyTorch tensors with dtype `torch.float32`.
    """
    return [torch.from_numpy(array).to(dtype=torch.float32) for array in arrays]


class BaseDynamicalDataset(abc.ABC, torch.utils.data.dataset.Dataset):
    """Abstract base class for dynamical system datasets."""

    def __init__(
        self,
        num_samples: int,
        signal_length: int,
        smnr_db: float,
        H: torch.Tensor,
    ) -> None:
        """Constructor.

        Args:
            num_samples: Number of samples to generate in the dataset.
            signal_length: Number of time steps in the signal.
            smnr_db: The Signal Measurement Noise Ratio in dB.
            H: State to observation matrix.
        """
        self._num_samples = num_samples
        self._signal_length = signal_length
        self._smnr_db = smnr_db
        self._H = H
        self._states, self._observations, self._noisy_observations, self._Cws = self._generate_data()
        # Expected dimension is 3: (num_samples, signal_length, dim).
        _check_same_shape_and_expected_dim(
            self._states,
            self._observations,
            self._noisy_observations,
            expected_dim=3
        )

    @abc.abstractmethod
    def _generate_data(
        self
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generates the dataset.

        Returns:
            Tuple of three tensors representing 'states', 'observations', and 'noisy observations'.
        """
        pass

    @property
    def H(self) -> torch.Tensor:
        """Returns the observation matrix."""
        return self._H

    def state_to_observation(
        self,
        states: torch.Tensor,
        device: str = 'cpu'
    ) -> torch.Tensor:
        """Maps states to observations by `H @ states`.

        Args:
            stats: The states to map.
            device: The device to use for the conversion, defaults to 'cpu'.

        Returns:
            The observations.
        """
        return torch.einsum(
            '...c,dc->...d',
            states.to(device),
            self._H.to(device)
        )

    def __len__(self) -> int:
        return self._num_samples

    def __getitem__(self, index: int):
        return self._states[index], self._observations[index], self._noisy_observations[index], self._Cws[index]

    def collate_fn(self, batch: Sequence) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns a batch consisting of tensors."""
    
        states, observations, noisy_observations, Cws = zip(*batch)
        states = torch.FloatTensor(np.array(states))
        observations = torch.FloatTensor(np.array(observations))
        noisy_observations = torch.FloatTensor(np.array(noisy_observations))
        Cws = torch.FloatTensor(np.array(Cws))
        return states, observations, noisy_observations, Cws


@gin.configurable
class LorenzAttractor(BaseDynamicalDataset):
    """Lorenz attractor dataset."""

    def __init__(self, num_samples: int, signal_length: int, smnr_db: float):
        """Constructor.

        Args:
            num_samples: Number of samples to generate in the dataset.
            signal_length: Number of time steps in the signal.
            smnr_db: The Signal Measurement Noise Ratio in dB.
        """
        self._lorenz_attractor_model = lorenz_attractor.LorenzSSM(delta_d=0.02)
        super().__init__(
            num_samples=num_samples,
            signal_length=signal_length,
            smnr_db=smnr_db,
            H=torch.eye(3),
        )
    
    def _generate_data(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate the double pendulum dataset."""
        all_states = []
        noisy_observations = []
        Cws = []
        for _ in range(self._num_samples):
            states, observations = self._lorenz_attractor_model.generate_single_sequence(
                self._signal_length, sigma_e2_dB=-10, smnr_dB=self._smnr_db,
            )
            Cw = np.expand_dims(
                self._lorenz_attractor_model.observation_cov,
                axis=0,
            )
            Cws.append(Cw)
            all_states.append(states)
            noisy_observations.append(observations)
        batched_states = np.stack(all_states, axis=0)
        observations = batched_states.copy()
        noisy_observations = np.stack(noisy_observations, axis=0)
        Cws = np.stack(Cws, axis=0)
        return _convert_arrays_to_tensors(
            batched_states,
            observations,
            noisy_observations,
            Cws,
        )


@gin.configurable
class ChenAttractor(BaseDynamicalDataset):
    """Chen attractor dataset."""

    def __init__(self, num_samples: int, signal_length: int, smnr_db: float):
        """Constructor.

        Args:
            num_samples: Number of samples to generate in the dataset.
            signal_length: Number of time steps in the signal.
            smnr_db: The Signal Measurement Noise Ratio in dB.
        """
        self._chen_attractor_model = lorenz_attractor.LorenzSSM(
            alpha=1.0, delta=0.002, delta_d=0.002 / 5, decimate=True,
        )
        super().__init__(
            num_samples=num_samples,
            signal_length=signal_length,
            smnr_db=smnr_db,
            H=torch.eye(3),
        )
    
    def _generate_data(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate the double pendulum dataset."""
        all_states = []
        noisy_observations = []
        Cws = []
        for _ in range(self._num_samples):
            states, observations = self._chen_attractor_model.generate_single_sequence(
                self._signal_length, sigma_e2_dB=-10, smnr_dB=self._smnr_db,
            )
            Cw = np.expand_dims(
                self._chen_attractor_model.observation_cov,
                axis=0,
            )
            Cws.append(Cw)
            all_states.append(states)
            noisy_observations.append(observations)
        batched_states = np.stack(all_states, axis=0)
        observations = batched_states.copy()
        noisy_observations = np.stack(noisy_observations, axis=0)
        Cws = np.stack(Cws, axis=0)
        return _convert_arrays_to_tensors(
            batched_states,
            observations,
            noisy_observations,
            Cws,
        )

@gin.configurable
class DobulePendulum(BaseDynamicalDataset):
    """Double Pendulum dataset."""

    def __init__(self, num_samples: int, signal_length: int, smnr_db: float):
        """Constructor.

        Args:
            num_samples: Number of samples to generate in the dataset.
            signal_length: Number of time steps in the signal.
            smnr_db: The Signal Measurement Noise Ratio in dB.
        """
        self._dp_constants = double_pendulum.DSPConstants(
            m1=1.2,
            m2=0.8,
            l1=1.2,
            l2=0.8,
            k1=500.0,
            k2=1000.0,
            process_noise_db=-50.0,
        )
        super().__init__(
            num_samples=num_samples,
            signal_length=signal_length,
            smnr_db=smnr_db,
            H=torch.eye(4),
        )
    
    def _generate_data(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate the double pendulum dataset."""
        states, noisy_observations, Cws = double_pendulum.gen_dataset(
            self._num_samples,
            self._signal_length,
            self._smnr_db,
            self._dp_constants,
            stochastic=True
        )
        return _convert_arrays_to_tensors(
            states,
            states.copy(),
            noisy_observations,
            np.expand_dims(Cws, axis=1),
        )


@gin.configurable
def get_dataloader(
    dataset: torch.utils.data.Dataset, batch_size: int, num_workers: int, shuffle: bool
) -> torch.utils.data.DataLoader:
    """Returns a dataloader of the dataset."""
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        collate_fn=dataset.collate_fn,
    )


def main():
    x, *_ = ClimateData(100, 100, 20)._generate_data()
    n = 30
    print(x[n, :, 0])
    print(x[n, :, 1])
    print(x[n, :, 2])

if __name__ == "__main__":
    main()