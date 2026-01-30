# Data-driven Nonlinear Smoother (DNS)

A deep learning framework for state estimation in model-free nonlinear dynamical systems.

## Overview

This repository implements a data-driven approach to nonlinear state smoothing for chaotic dynamical systems. The model combines:
- Causal and anti-causal CNNs for temporal feature extraction
- Bidirectional GRU networks for forward-backward processing
- Bayesian filtering framework for posterior estimation

**Supported Systems:**
- Lorenz Attractor
- Chen Attractor  
- Double Spring Pendulum

## Installation
```bash
git clone https://github.com/fcumlin/DataDrivenNonLinearSmoother.git
cd DataDrivenNonLinearSmoother
pip install torch numpy matplotlib gin-config tqdm
```

## Quick Start

Train on the Lorenz attractor:
```bash
bash launch.sh
```

Or train on a specific system:
```bash
python train.py --gin_path "configs/lorenz.gin" --save_path "runs/lorenz"
python train.py --gin_path "configs/chen.gin" --save_path "runs/chen"
python train.py --gin_path "configs/double_pendulum.gin" --save_path "runs/double_pendulum"
```

## Project Structure
```
├── configs/              # Gin configuration files
│   ├── lorenz.gin
│   ├── chen.gin
│   └── double_pendulum.gin
├── dataset.py           # Dataset generators and loaders
├── model.py             # DNS neural network architecture
├── train.py             # Training loop and evaluation
├── utils.py             # Loss accumulators and monitoring
├── lorenz_attractor.py  # Lorenz/Chen attractor simulator
├── double_pendulum.py   # Double spring pendulum simulator
└── launch.sh            # Training script
```

## Configuration

All hyperparameters are managed via Gin configs. Key parameters:
```gin
# Model architecture
DatadrivenNonlinearSmoother:
    input_dim = 3
    hidden_dim = 30
    output_dim = 3
    num_layers = 1

# Training
TrainingLoop:
    num_epochs = 200
    learning_rate = 1e-3
    batch_size = 64
    device = 'cuda'
```

## Model Architecture

The DNS model consists of:

1. **Preprocessing**: Causal/anti-causal 1D CNNs extract temporal features
2. **Encoder**: Bidirectional GRU processes forward and backward sequences
3. **Decoder**: Dense layers predict state mean and covariance
4. **Refinement**: Secondary GRU refines predictions using posterior estimates

## Results

Training outputs are saved to `runs/<s>/`:
- Model checkpoints (`danse_model_state_dict_best.pt`)
- Training logs (`train.log`)
- Validation plots (trajectory visualizations per dimension)

## Performance Metrics

The model is evaluated using:
- **NMSE (dB)**: Normalized Mean Squared Error in decibels
- **Log Likelihood**: Goodness of fit for posterior distributions

## Custom Datasets

To add a new dynamical system:

1. Implement a class inheriting from `BaseDynamicalDataset` in `dataset.py`
2. Override `_generate_data()` to return states and observations
3. Create a corresponding `.gin` config file
4. Register the dataset class with `@gin.configurable`

Example:
```python
@gin.configurable
class NewSystem(BaseDynamicalDataset):
    def _generate_data(self):
        # Generate your system's trajectories
        return states, observations, noisy_observations, Cws
```

## Citation

If you use this code in your research, please cite:
```bibtex
@inproceedings{cumlin2026dns,
  author = {Cumlin, Fredrik and Ghosh, Anubhab and Chatterjee, Saikat},
  title = {{DNS}: Data-Driven Nonlinear Smoother for Complex Model-Free Process},
  booktitle = {ICASSP 2026 - 2026 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  year = {2026},
  pages = {TBD},
  organization = {IEEE},
  address = {TBD},
  doi = {TBD}
}
```

## License

MIT License - see LICENSE file for details

## Contact

For questions or issues, contact: fcumlin@gmail.com

## Acknowledgments

This implementation uses:
- [PyTorch](https://pytorch.org/) for deep learning
- [Gin-config](https://github.com/google/gin-config) for configuration management
