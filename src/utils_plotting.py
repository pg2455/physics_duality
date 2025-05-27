import matplotlib.pyplot as plt
import pathlib
import numpy as np

def plot_optimization_analysis(grads, all_losses, non_reg_losses, det_hessian=None, trace_hessian=None, N=None, beta_target=None, output_path=None):
    """
    Creates a figure showing gradient, loss and Hessian analysis plots.
    
    Args:
        grads: List of gradient magnitudes
        all_losses: List of total losses
        non_reg_losses: List of non-regularized losses
        det_hessian: Optional list of Hessian determinants
        trace_hessian: Optional list of Hessian traces
        N: System size
        beta_target: Target beta value
        output_path: Path to save the figure
    """
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    fig.tight_layout(pad=3.0)

    # Plot gradients
    ax = axs[0, 0]
    ax.plot(grads, label='gradient')
    ax.set_ylabel('Gradient magnitude')
    ax.set_xlabel('Iteration')
    ax.legend()
    ax.set_title('Gradient')

    # Plot loss again for comparison
    ax = axs[0, 1] 
    ax.plot(all_losses, label='loss')
    if isinstance(non_reg_losses, dict):
        ax.plot(non_reg_losses['base_loss'], label='base_loss', alpha=0.5)
        ax.plot(non_reg_losses['mmd_loss'], label='mmd_loss', alpha=0.5)
    else:
        ax.plot(non_reg_losses, label='non-reg-loss', alpha=0.5)
    ax.set_ylabel('Loss')
    ax.set_xlabel('Iteration')
    ax.legend()
    ax.set_title('Losses')

    # Plot determinant of Hessian if available
    ax = axs[1, 0]
    if det_hessian is not None:
        ax.plot(det_hessian, label='det(H)')
        ax.set_ylabel('Determinant')
        ax.set_xlabel('Iteration')
        ax.legend()
    ax.set_title('Determinant of Hessian')

    # Plot trace of Hessian if available  
    ax = axs[1, 1]
    if trace_hessian is not None:
        ax.plot(trace_hessian, label='tr(H)')
        ax.set_ylabel('Trace')
        ax.set_xlabel('Iteration')
        ax.legend()
    ax.set_title('Trace of Hessian')

    title = f"Gradient and Hessian Analysis (N={N}, beta={beta_target})"
    fig.suptitle(title)
    
    if output_path is not None:
        fig.savefig(str(pathlib.Path(output_path) / "grad_hessian_plot.png"))
    
    return fig


def plot_training_progress(betas, best_betas, beta_target_dual, all_losses, o_mappings, o_mappings_5, features_model, features_data, attn_weights, best_selected_dim, N, N_COMPONENTS, beta_target, config):
    """
    Creates a figure showing training progress plots.
    
    Args:
        betas: List of beta values during training
        best_betas: List of best beta values found
        beta_target_dual: Target beta value in dual space
        all_losses: List of total losses
        o_mappings: Dictionary of observable mappings
        o_mappings_5: Dictionary of observable mappings for index 5
        features_model: List of model features
        features_data: List of data features  
        attn_weights: Array of attention weights
        best_selected_dim: Best selected dimension
        N: System size
        N_COMPONENTS: Number of components
        beta_target: Target beta value
        config: Configuration dictionary
    """
    fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(10,10), dpi=100)

    ax = axs[0, 0]
    if N_COMPONENTS == 2:
        ax.plot([x[0] for x in betas], label='beta', color='blue')
        ax.plot([x[1] for x in betas], label='K', color='black')
        ax.plot([x[0] for x in best_betas], label='beta', color='blue', linestyle=':')
        ax.plot([x[1] for x in best_betas], label='K', color='black', linestyle=':')
        ax.hlines(np.mean([x[0] for x in betas][-100:]), 0, len(betas), linestyle='--', color='blue')
        ax.hlines(np.mean([x[1] for x in betas][-100:]), 0, len(betas), linestyle='--', color='black')
    else:
        ax.plot(betas, label='beta', color='blue')
        ax.plot(best_betas, label='beta', color='blue', linestyle=':')
        ax.hlines(np.mean(betas[-100:]), 0, len(betas), linestyle='--', color='blue')

    ax.hlines(beta_target_dual[0], 0, len(betas), color='green', linestyle='--')
    ax.set_ylabel('beta, K')
    ax.set_title("beta, K")

    ax = axs[0, 1]
    ax.plot(all_losses)
    ax.set_ylabel('MMD loss')
    ax.hlines(+0, 0, len(betas), linestyle='--', color='r')
    ax.set_ylabel('loss')

    ax = axs[1, 0]
    ax.plot(o_mappings[-1], label="Observable(Model_selected) on -1", color='green')
    ax.plot(o_mappings[1], label="Observable(Model_selected) on +1", color='red')
    ax.hlines(np.exp(2 * beta_target_dual[0]), 0, len(betas), linestyle='--', color='green')
    ax.hlines(np.exp(-2 * beta_target_dual[0]), 0, len(betas), linestyle='--', color='red')
    ax.legend()
    ax.set_ylabel("Mapping of observable")
    ax.set_title(f"Selected Index: {best_selected_dim}")

    ax = axs[1, 1]
    ax.plot(np.array(features_model)-np.array(features_data), label='delta', alpha=0.5)
    ax.hlines(+0, 0, len(betas), linestyle='--', color='r')
    ax.set_ylabel('feature difference')

    ax = axs[2, 0]
    bottom = np.zeros(attn_weights.shape[0])

    colors = ['black', 'green', 'red', 'orange', 'pink', 'purple', 'blue']

    for idx in range(7):
        ax.bar(np.arange(attn_weights.shape[0]), attn_weights[:, idx], bottom=bottom, label=f'{idx}', color=colors[idx])
        bottom += attn_weights[:, idx]

    ax.legend()
    ax.set_title('Attn Weights')

    ax = axs[2, 1]
    x = - attn_weights * np.log(attn_weights)
    ax.plot(np.sum(x, axis=1))
    ax.set_title('Attn Entropy')

    title = f"N={N} N_COMPONENTS={N_COMPONENTS} BETA={beta_target} beta_lr={config.beta_lr} OptType:{config.beta_optimize_type}"
    fig.suptitle(title)
    fig.savefig(str(pathlib.Path(config.output_path) / "plot.png"))
    
    return fig