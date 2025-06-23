# A Machine Learning Approach to Duality in Statistical Physics (ICML 2025)

This repository accompanies the paper published at ICML 2025, titled [A Machine Learning Approach to Duality in Statistical Physics](https://openreview.net/forum?id=bWeLpOgqGp).


## 2. Requirements & Installation

- Python 3.10+
- PyTorch
- numpy
- matplotlib
- addict
- pyyaml

Install dependencies:
```bash
pip install torch numpy matplotlib addict pyyaml
```

## 3. Quick Start

**Run with default configuration:**
```bash
python main.py
```

**Run with a custom configuration:**

```bash
python main.py config=configs/my_experiment.yaml
```

You can edit `configs/default.yaml` or create your own YAML config file.


**Overwrite configuration through command line:**

You can overwrite configuration parameters by passing them through command line. 
```bash
python main.py config=configs/default.yaml N=4
```

## 4. Configuration

Below are the configuration options actually used in `main.py`. See `configs/default.yaml` for a full example.

```yaml
# Core configuration
N: 8                        # Size of the lattice
N_COMPONENTS: 2             # Number of terms in Hamiltonian
beta_target: 0.2            # Target beta value
K_target: 0.1               # Target K value (when N_COMPONENTS = 2)
beta_model: 0.1             # Initial beta value for model
K_model: 0.1                # Initial K value for model
alpha: 0.1                  # Averaging parameter
CD_steps: 1000              # Number of Contrastive Divergence steps
num_samples: 100            # Number of samples to generate
sample_multiplier: 10       # Multiplier for sample rate
seed: 42                    # Random seed
output_dir: "./output"      # Base output directory
output_path: null           # (Optional) Full output path (overrides output_dir)

# Mapping
map_type: "softattn"        # Type of mapping
learn_map_function: true    # Whether to learn mapping function
map_index: "soft"           # Mapping index type ("original", "dual", "soft")
discretize_attn: false      # Whether to discretize attention
random_init: false          # Whether to randomly initialize mapping
all_lr: 0.01                # Learning rate for mapping and beta
attention_temperature: 0.5  # Temperature for attention
attn_init_std: 0.1          # Std for attention initialization

# Optimization
beta_optimize_type: "manual"    # "in-built" or "manual"
beta_learning_mask: "1,0"       # Mask for beta learning
epsilon_hessian: 1e-6           # Epsilon for Hessian regularization
early_stopping_steps: 200       # Steps for early stopping
clip_gradients: false           # Whether to clip gradients

# Optional/Advanced
original_frame_penalty: 0.1     # Penalty for original frame
bias_attn: false                # Whether to bias attention
link_feature_order: 2           # Feature order for links

# Logging
logfile: null                   # Path to log file (auto-set)
```

## 5. Output & Results

Results, logs, and plots are saved in the output directory specified by `output_dir` or `output_path`.

## 6. Contact

For questions or issues, please open an issue on the repository.

## 7. How to Cite

If you use this code or ideas from our paper, please cite:

```bibtex
@inproceedings{icml2025duality,
  title={A Machine Learning Approach to Duality in Statistical Physics},
  author={Gupta, Prateek and Ferrari, Andrea E.V. and Iqbal, Nabil},
  booktitle={Proceedings of the 42nd International Conference on Machine Learning (ICML)},
  year={2025},
  url={https://openreview.net/forum?id=bWeLpOgqGp}
}
```

