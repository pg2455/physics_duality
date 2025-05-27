import pathlib
import random
import numpy as np
import yaml
import json
import math
import multiprocessing as mp
import matplotlib.pyplot as plt
import time
from addict import Dict

import sys
sys.path.append('src/')
import utils_ising, utils_exp, kernel, utils_mask, utils
# import utils_mask_eff
import utils_plotting
import torch
import torch.nn as nn
import torch.nn.functional as F


if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using GPU:", device)
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS:", device)
else:
    device = torch.device("cpu")
    print("No GPU -> using CPU:", device)

import utils_exp

ROOT = pathlib.Path(__file__).resolve().parent


class G_plaquette_softattn(nn.Module):
    def __init__(self, n_dims=7, temperature=0.5, random_init=False, discretize_attn=False, map_index="soft", learn_map_function=True):
        super().__init__()
        self.f = nn.Linear(1, 1, bias=True)
        if random_init:
            nn.init.uniform_(self.f.weight, -1.0, 1.0) # TODO: Use Uniform Sampling
            nn.init.uniform_(self.f.bias, -1.0, 1.0)
            print("Using Uniform initialization:")
        else:
            self.f.weight.data = -0.25 * torch.ones_like(self.f.weight.data).float()
            self.f.bias.data = 0.75 * torch.ones_like(self.f.bias.data).float()
            print("Using default initialization:")
        print(f"Weight: {self.f.weight.data.item():.4f}")
        print(f"Bias: {self.f.bias.data.item():.4f}")

        self.embedding = nn.init.normal_(nn.Parameter(torch.zeros(n_dims)), mean=0.0, std=config.get("attn_init_std", 0.1))
        self.embedding.data[0] = self.embedding.data[0] - (1 if config.get("bias_attn", False) else 0)
        print("Embedding intialized with", self.embedding.data.cpu())
        self.initial_temperature = temperature
        self.min_temperature = 0.1
        self.temperature = self.initial_temperature
        self.discretize_attn = discretize_attn
        self.map_index = map_index
        self.learn_map_function = learn_map_function
    
    def forward(self, x, beta=None):
        if self.map_index == "dual":
            # Force selection of index 2
            attn_weights = torch.zeros_like(self.embedding)
            attn_weights[2] = 1.0
        elif self.map_index == "original":
            # Force selection of index 6
            attn_weights = torch.zeros_like(self.embedding)
            attn_weights[6] = 1.0
        else:
            # Original soft attention logic
            attn_weights = F.gumbel_softmax(self.embedding, tau=self.temperature, hard=self.discretize_attn)
        
        x = torch.sum(x * attn_weights[None, None], dim=-1, keepdims=True)
       
        if self.learn_map_function:
            return self.f(x)
        else:
            if self.map_index == "dual":
                assert beta is not None, "beta can't be none"
                b = torch.tensor(beta[0], device=x.device)
                return torch.where(x == 1, torch.exp(-2 * b), torch.exp(2 * b))
            elif  self.map_index == "original":
                return x
            else:
                raise ValueError("not learning a map function is not an option")

    def get_indices(self):
        attn = F.gumbel_softmax(self.embedding, tau=self.temperature, hard=True)
        return torch.where(attn)[0]

    def get_attn_weights(self, temperature=None):
        if temperature is None:
            temperature = self.temperature
        return F.gumbel_softmax(self.embedding, tau=temperature, hard=False)

    def update_temperature(self, epoch, total_epochs):
        new_temp = self.initial_temperature - (self.initial_temperature - self.min_temperature) * (epoch / total_epochs)
        self.temperature = max(self.min_temperature, new_temp)


def update_beta_component(feature_mapped, running_data_mean, loss_vec, alpha,
                         avg_H_terms_model, H_terms_model, beta_model, 
                         beta_learning_mask, last_beta, config):
    """Update single beta component."""
    # uses control variates
    loss_vec = (feature_mapped - running_data_mean[None]).detach().cpu().numpy() # [B, N_FEATURES]
    x = avg_H_terms_model[None, ...] - H_terms_model # B, N_COMPONENTS
    weighted_loss_vec = x[..., None] * loss_vec[:, None, :] # [B, N_COMPONENTS, 1] x [B, 1, N_FEATURES] --> [B, N_COMPONENTS, N_FEATURES]
    grad_components = loss_vec.mean(0)[None] * weighted_loss_vec.mean(0) # [1, N_FEATURES] x [N_COMPONENTS, N_FEATURES] --> [N_COMPONENTS, N_FEATURES]
    grad = 2 * grad_components.mean(1) # [N_COMPONENTS] # Factor of 2 to make sure theta and maps are learned at the same rate. 

    # Hessian calculation (NOTE: some confusion regarding the sign of x2; -x2 works as expected)
    x2 = np.power(avg_H_terms_model, 2) - np.power(H_terms_model, 2).mean(0) # N_COMPONENTS
    loss_weighted_sq = (np.power(x[..., None], 2) * loss_vec[:, None, :]).mean(0) #  [B, N_COMPONENTS, 1] x [B, 1, N_FEATURES] --> [N_COMPONENTS, N_FEATURES]
    variance_term = -x2[..., None] * loss_vec.mean(0)[None] # [N_COMPONENTS, 1] x [1, N_FEATURES] --> [N_COMPONENTS, N_FEATURES]
    double_derivative = loss_weighted_sq + variance_term # [N_COMPONENTS, N_FEATURES]
    double_derivative = double_derivative.mean(1) # N_COMPONENTS 

    y = avg_H_terms_model[0] * avg_H_terms_model[1]  # 1 x 1
    y += -(H_terms_model[:, 0] * H_terms_model[:, 1]).mean(0) # B x B --> 1
    covariance_term = loss_vec.mean(0) * y # N_FEATURES
    loss_weighted_term = (x[:, 0, None] * x[:, 1, None] * loss_vec).mean(0) # [B, 1] x [B, 1] x [B, N_FEATURES] --> [N_FEATURES]
    cross_double_derivative = (covariance_term + loss_weighted_term).mean() # 1

    a = double_derivative[0]
    d = double_derivative[1]
    b = c = cross_double_derivative
    hessian = np.array([[a, b], [c, d]])
    H_reg = hessian + config.get("epsilon_hessian", 1e-6) * np.eye(2)
    inverse_hessian = np.linalg.inv(H_reg)


    if torch.where(beta_learning_mask == 1)[0].size  == 2:
        det_h = np.linalg.det(H_reg)
        trace_h = np.trace(H_reg)
    elif torch.where(beta_learning_mask == 1)[0][0] == 0: # beta is being learned
        trace_h = det_h = a 
    else:
        trace_h = det_h = d

    # Update only the selected component
    if config.beta_optimize_type == "in-built":
        grad = grad * beta_learning_mask.numpy()
    elif config.beta_optimize_type == "manual":
        if torch.where(beta_learning_mask == 1)[0].size  == 2:
            grad = inverse_hessian @ grad
        elif torch.where(beta_learning_mask == 1)[0] == 0:
            grad = grad / (a + 1e-8)
        else:
            grad = grad / (d + 1e-8)
        grad = grad * beta_learning_mask.numpy()
        
    return beta_model, grad, det_h, trace_h



def find_beta(beta_target: int, N: int, config):
    """
    Finds beta that closely matches the samples generated at beta_target.
    """

    N_COMPONENTS = config.N_COMPONENTS # number of terms in hamiltonian

    if beta_target == 0:
        beta_target_dual = 0
    else:
        beta_target_dual = utils_ising.get_dual(beta_target)

    if N_COMPONENTS == 1:
        beta_target = np.array([beta_target]) # beta1, beta2, ...
        beta_model = torch.nn.Parameter(torch.tensor([config.beta_model]), requires_grad=True)
        beta_target_dual = np.array([beta_target_dual])
    elif N_COMPONENTS == 2:
        beta_target = np.array([beta_target, config.K_target]) # beta1, beta2, ...
        beta_model = torch.nn.Parameter(torch.tensor([config.beta_model, config.K_model]), requires_grad=True)
        beta_target_dual = np.array([beta_target_dual, 0])
    else:
        raise ValueError(f"Unrecognized N_COMPONENTS: {N_COMPONENTS}")

    # beta learning mask
    beta_learning_mask_str = config.get("beta_learning_mask", "1,0")
    beta_learning_mask = [float(x) for x in beta_learning_mask_str.split(",")]
    beta_learning_mask = torch.tensor(beta_learning_mask)

    # masking beta
    beta_model.data = beta_model.data * beta_learning_mask
    beta_target = beta_target * beta_learning_mask.numpy()

    # gather the edges for indexing
    i, j = utils_ising.get_edges(rows=N, cols=N)
    edge_i, edge_j, ij_to_edge_matrix_idxs = utils_ising.get_edge_idxs_with_map(
        N, i, j
    )
    indexing_link_to_edge_matrix_mapping, _ = utils_ising.get_edge_matrix_indexers_for_mapping(
        N, i, j, ij_to_edge_matrix_idxs
    )
    if config.get('link_feature_order', 2) == 2:
        indexing_link_to_edge_matrix_mapping = utils_ising.get_features_with_second_order_links(
            N, i, j, edge_i, edge_j, ij_to_edge_matrix_idxs, 
            indexing_link_to_edge_matrix_mapping
        )
    indexers = torch.tensor(indexing_link_to_edge_matrix_mapping, device=device)
    mapping_edge_matrix_row_indices = indexers[:, :, 0]
    mapping_edge_matrix_col_indices = indexers[:, :, 1]

    # masks for featurization
    masks = utils_ising.generate_masks(N)
    masks = masks.to(device, dtype=torch.uint8)

    # learning related params
    alpha = config.alpha
    CD_steps = config.CD_steps

    # sampling parameters
    num_samples = config.num_samples
    sample_rate = N * N * config.sample_multiplier
    
    # mapping
    if config.map_type == "softattn":
        observable_map = G_plaquette_softattn(
            n_dims=indexers.shape[1],
            learn_map_function=config.get("learn_map_function", True), 
            map_index=config.get("map_index", "soft"),
            discretize_attn=config.get("discretize_attn", False),
            random_init=config.get("random_init", False),
            temperature=config.get('attention_temperature', 
                                 0.5 if indexers.shape[1]==7 else 1.0),
            )
    else:
        raise ValueError(f"Unrecognized map_type: {config.map_type}")


    # mapping optimizer
    observable_map.to(device)
    optimizer = torch.optim.Adam(list(observable_map.parameters()) + [beta_model], config.get('all_lr', 0.01))

    betas, best_betas = [beta_model.detach().cpu().numpy()], [beta_model.detach().cpu().numpy()]
    all_losses, max_diffs, grads, diffs = [], [], [], []
    selected_dimensions = []
    features_data, features_model, non_reg_losses = [], [], {'base_loss':[]}
    features_data_saved, features_model_saved = [], []
    o_mappings, o_mappings_5, attn_weights = {1:[], -1:[]}, {1:[], -1:[]}, []
    best_total_loss, no_improvement_count, best_selected_dim = np.inf, 0, -1
    det_hessian, trace_hessian = [], []
    running_data_mean, running_data_magnetization = 0, 0
    n_links = masks[0, 0, :, :].sum(dim=-1)
    rng = np.random.default_rng(seed=config.seed)
    beta_learning_mask_tmp = beta_learning_mask.clone()
    for cd_step in range(CD_steps+1):

        ## Sample data
        args = [(N, beta_target, i, j, edge_i, edge_j, num_samples, sample_rate, np.random.default_rng(seed=rng.integers(2**32 - 1)))]
        args += [(N, beta_model.detach().cpu().numpy(), i, j, edge_i, edge_j, num_samples, sample_rate, np.random.default_rng(seed=rng.integers(2**32 - 1)))]
        with mp.Pool(processes=min(4, len(args))) as pool:
            results = pool.starmap(utils_ising.sample_stats_edge_wrapper, args)

        o_data, *_ = results[0]
        _, H_terms_model, avg_H_terms_model, edge_matrices = results[1]

        o_data, edge_matrices = o_data.to(device), edge_matrices.to(device)
        link_mapped = edge_matrices[:, mapping_edge_matrix_row_indices, mapping_edge_matrix_col_indices]
        o_mapped = observable_map(link_mapped).squeeze()

        # compute the loss -- this has dimensions of num_samples x num_features
        feature_data = utils_ising.feature_samplewise(masks, o_data)
        feature_mapped = utils_ising.feature_samplewise(masks, o_mapped)

        running_data_mean = (feature_data.mean(0) + running_data_mean * cd_step) / (cd_step + 1)

        data_target = running_data_mean 
        data_model = feature_mapped       

        loss_vec = data_model.mean(0) - data_target# [B, N_FEATURES]
        base_loss = torch.mean(loss_vec ** 2)
        non_reg_losses['base_loss'].append(base_loss.clone().detach().cpu().item())

        loss = base_loss + config.get("original_frame_penalty", 0) * F.softmax(observable_map.embedding, dim=-1)[6]

        ## Backpropagate
        optimizer.zero_grad()
        loss.backward()
        
        # compute beta gradient
        grad = np.zeros(1)
        beta_model, grad, det_h, trace_h = update_beta_component(
            data_model, data_target, loss_vec, alpha,
            avg_H_terms_model, H_terms_model, beta_model,
            beta_learning_mask_tmp, betas[-1], config
        )
        beta_model.grad = torch.tensor(grad, dtype=beta_model.dtype, device=beta_model.device)
        

        det_hessian.append(det_h)
        trace_hessian.append(trace_h)
    
        if config.get("clip_gradients", False):
            torch.nn.utils.clip_grad_norm_(
                list(observable_map.parameters()) + [beta_model], 
                max_norm=1.0)
        optimizer.step()

        beta_model.data = (1-alpha) * beta_model.data + alpha * torch.tensor(betas[-1])

        # record data
        betas.append(beta_model.clone().detach().cpu().numpy())
        all_losses.append(loss.detach().cpu().item())
        diffs.append(np.abs(betas[-1] - beta_target_dual))
        max_diffs.append(diffs[-1].max())
        grads.append(np.sqrt((grad ** 2).sum()).item())
        features_data_saved.append(feature_data.detach().cpu().numpy())
        features_model_saved.append(feature_mapped.detach().cpu().numpy())

        features_data.append(utils_ising.feature_eval(masks, o_data).detach().cpu().numpy())
        features_model.append(utils_ising.feature_eval(masks, o_mapped).detach().cpu().numpy())

        # dimension selection
        with torch.no_grad():
            out = observable_map.get_indices()
            selected_dimensions.append(out.cpu().tolist())

        # if beta_target[1] == 0: # base case
        with torch.no_grad():
            selected_index = observable_map.get_indices().cpu().item()
            for focus_idx, map_dict in [(5, o_mappings_5), (selected_index, o_mappings)]:
                dummy_input = torch.ones((2, link_mapped.shape[-1]), device=device)
                dummy_input[0, focus_idx] = -1 
                dummy_input[1, focus_idx] = 1
                dummy_out = observable_map(dummy_input.unsqueeze(0)).squeeze().cpu()
                map_dict[-1].append(dummy_out[0].item())
                map_dict[1].append(dummy_out[1].item())

        attn_weights.append(observable_map.get_attn_weights().detach().cpu().tolist())

        # early stopping
        if all_losses[-1] < best_total_loss:
            best_total_loss = all_losses[-1]
            best_beta_model = betas[-1]
            best_selected_dim = selected_dimensions[-1]
            no_improvement_count = 0
            if observable_map is not None:
                torch.save(observable_map.state_dict(), pathlib.Path(config.output_path) / "mapping_best_params.pkl")
            utils_exp.log(f"Best loss found so far: {best_total_loss: 1.4e}. Grad: {grads[-1]: 0.4f}. Dims: {selected_dimensions[-1]}.", config.logfile)
        
        else:
            no_improvement_count += 1
            
            if no_improvement_count % 100 == 0:
                optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] * 0.5
                utils_exp.log(f"reducing learning rate to {optimizer.param_groups[0]['lr']}", config.logfile)
            
            if no_improvement_count % config.get("early_stopping_steps", 200) == 0:
                utils_exp.log("No improvement in kernel observed in the last 200 steps. Stopping.", config.logfile)
                break
        
        best_betas.append(best_beta_model)

        # logs
        if cd_step % 100 == 0:
            string = f"@ CD Step: {cd_step}\tLoss: {np.mean(all_losses[-100:]): 1.4e}"
            string += f"\tbeta_grad: {np.abs(grad).max(): 0.3f}\tbest_total_loss: {best_total_loss: 1.4e}"
            utils_exp.log(string, config.logfile)

    attn_weights = np.array(attn_weights)
    
    # store results
    output_path = pathlib.Path(config.output_path) /  "results.json"    
    to_save = Dict()
    to_save.training_data = dict(
        filepath=str(output_path),
        betas = [x.tolist() for x in betas],
        best_betas = [x.tolist() for x in best_betas],
        o_mappings = o_mappings,
        all_losses = all_losses,
        o_mappings_5 = o_mappings_5,
        attn_weights = attn_weights.tolist(),
        non_reg_losses = non_reg_losses,
        best_selected_dim = best_selected_dim,
        grads = grads,
        features_data = [x.tolist() for x in features_data_saved],
        features_model = [x.tolist() for x in features_model_saved],
    )
    to_save.update(config)

    to_save = utils.post_process_results(to_save)
    with open(output_path, "w") as f:
        json.dump(to_save, f)
    
    # plot training progress
    utils_plotting.plot_training_progress(
        betas=betas,
        best_betas=best_betas, 
        beta_target_dual=beta_target_dual,
        all_losses=all_losses,
        o_mappings=o_mappings,
        o_mappings_5=o_mappings_5,
        features_model=features_model,
        features_data=features_data,
        attn_weights=attn_weights,
        best_selected_dim=best_selected_dim,
        N=N,
        N_COMPONENTS=N_COMPONENTS,
        beta_target=beta_target,
        config=config
    )

    # plot optimization analysis
    utils_plotting.plot_optimization_analysis(
        grads=grads,
        all_losses=all_losses, 
        non_reg_losses=non_reg_losses,
        det_hessian=det_hessian,
        trace_hessian=trace_hessian,
        N=N,
        beta_target=beta_target,
        output_path=config.output_path
    )


def make_output_dir(config: Dict):
    """
    Establishes a suitable output directory name.
    """
    N, beta_target = config.N, config.beta_target

    # DIRECTORY NAMING
    output_dir  = pathlib.Path(config.output_dir).resolve()
    output_dir = output_dir / f"N-{N}-beta_target-{beta_target}" 
    output_dir = output_dir / f"K-start-{config.K_model}-beta_start-{config.beta_model}-seed-{config.seed}"

    n_tries = 0
    while True:
        output_dir = utils_exp.iterate_until_unique(output_dir)
        try:
            output_dir.mkdir(parents=True)
            break
        except Exception as e:
            print(e)
            print("Error creating output directory. Trying again...")
            n_tries += 1
            if n_tries > 10:
                raise ValueError("Failed to create output directory after 10 tries.")

        time.sleep(0.5) # try again in case the error is due to race condition
        continue
    return output_dir


if __name__ == "__main__":
    xargs = utils_exp.parse_and_load_config()

    if xargs.get("config", None) is None:
        with open(ROOT / 'configs/default.yaml') as f:
            config = Dict(yaml.safe_load(f))
    else:
        config = Dict(yaml.safe_load(open(xargs.config)))

    # overwrite config with command line arguments
    config.update(xargs)


    # Output path
    if config.get("output_path", None) is None:
        running_dir = make_output_dir(config)
    else:
        running_dir = pathlib.Path(config.output_path).resolve()
    config.output_path = str(running_dir)

    # logging
    config.logfile = str(running_dir / "log.txt")
    random.seed(config.seed)
    utils_exp.save_config(config)

    N, beta_target = config.N, config.beta_target
    find_beta(beta_target, N, config)