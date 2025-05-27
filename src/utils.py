import torch
import torch.nn as nn
import utils_ising
import numpy as np

def model_init(model, init_type='kaiming_normal'):
    """Initializes model parameters."""
    for p in model.parameters():
        if p.dim() > 1:
            if init_type == 'kaiming_normal':
                nn.init.kaiming_normal_(p)
            elif init_type == 'xavier_normal':
                nn.init.xavier_normal_(p)
            elif init_type == 'kaiming_uniform':
                nn.init.xavier_uniform_(p)
            elif init_type == 'xavier_uniform':
                nn.init.xavier_uniform_(p)

            else:
                raise ValueError(f"Undefined init_type:{init_type}")
            
    return model


def compute_gradient_norm(model):
    """Returns the gradient norm of the model."""
    total_norm = 0
    for p in model.parameters():
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm.item() ** 2
    return total_norm ** (1. / 2)

    

def compute_norm(model):
    """Returns the gradient norm of the model."""
    total_norm = 0
    for p in model.parameters():
        param_norm = p.data.norm(2)
        total_norm += param_norm.item() ** 2
    return total_norm ** (1. / 2)

    
def post_process_results(results, convert_np_arrays_to_list=True):
    """Post-processes the results."""
    IGNORE_KEYS = ['output_dir', 'output_path', 'logfile', 'git_commit']

    beta_dual = utils_ising.get_dual(results['beta_target']) if results['beta_target'] != 0 else 0
    results['beta_dual'] = beta_dual
    results['optimal_o_minus_1'] = np.exp(2 * beta_dual)
    results['optimal_o_plus_1'] = np.exp(-2 * beta_dual)
    results['filepath'] = results['filepath']

    # get optimal values
    results['final_beta'] = results['training_data']['best_betas'][-1][0]
    results['final_k'] = results['training_data']['best_betas'][-1][1]
    results['final_loss'] = np.mean(results['training_data']['all_losses'][-100:])
    
    # get optimal values
    i_star = np.argmin(results['training_data']['all_losses'])
    results['min_loss_idx'] = i_star
    results['min_loss'] = results['training_data']['all_losses'][i_star]
    if '-1' in results['training_data']['o_mappings']:
        results['o_mappings_minus_1'] = results['training_data']['o_mappings']['-1'][i_star]
    else:
        results['o_mappings_minus_1'] = results['training_data']['o_mappings'][-1][i_star]
    
    if '1' in results['training_data']['o_mappings']:
        results['o_mappings_plus_1'] = results['training_data']['o_mappings']['1'][i_star]
    else:
        results['o_mappings_plus_1'] = results['training_data']['o_mappings'][1][i_star]

    results['attn_weights_at_min_loss'] = results['training_data']['attn_weights'][i_star]
    
    # get diffs
    results['if_dual_mapping_minus_1_diff_abs'] = abs(results['o_mappings_minus_1'] - results['optimal_o_minus_1'])
    results['if_dual_mapping_plus_1_diff_abs'] = abs(results['o_mappings_plus_1'] - results['optimal_o_plus_1'])
    results['if_dual_mapping_all_diff_abs'] = results['if_dual_mapping_minus_1_diff_abs'] + results['if_dual_mapping_plus_1_diff_abs']
    results['if_dual_beta_diff_abs'] = abs(results['final_beta'] - results['beta_dual'])
    results['if_dual_total_diff_abs'] = results['if_dual_beta_diff_abs'] + results['if_dual_mapping_all_diff_abs']

    results['if_original_mapping_minus_1_diff_abs'] = abs(results['o_mappings_minus_1'] - -1)
    results['if_original_mapping_plus_1_diff_abs'] = abs(results['o_mappings_plus_1'] - 1)
    results['if_original_mapping_all_diff_abs'] = results['if_original_mapping_minus_1_diff_abs'] + results['if_original_mapping_plus_1_diff_abs']
    results['if_original_beta_diff_abs'] = min(abs(results['final_beta'] - results['beta_target']), abs(results['final_beta'] + results['beta_target']))
    results['if_original_total_diff_abs'] = results['if_original_beta_diff_abs'] + results['if_original_mapping_all_diff_abs']

    results['min_beta_diff_abs'] = min(results['if_dual_beta_diff_abs'], results['if_original_beta_diff_abs'])
    results['frame_with_min_beta_diff_abs'] = "dual" if results['if_dual_beta_diff_abs'] == results['min_beta_diff_abs'] else "original"
    if results['frame_with_min_beta_diff_abs'] == "dual":
        results['min_mapping_diff_abs'] = results['if_dual_mapping_all_diff_abs']
        results['if_original_frame_type'] = None
        results['min_total_diff_abs'] = results['if_dual_total_diff_abs']
    else:
        results['min_mapping_diff_abs'] = results['if_original_mapping_all_diff_abs']
        results['if_original_frame_type'] = "antiferromagnetic" if results['final_beta'] < 0 else "ferromagnetic"
        results['min_total_diff_abs'] = results['if_original_total_diff_abs']

    results['selected_dim'] = results['training_data']['best_selected_dim'][0]
    if results['selected_dim'] in [2, 5]:
        results['frame_by_selected_dim'] = 'dual'
    elif results['selected_dim'] == 6:
        results['frame_by_selected_dim'] = 'original'
    else:
        results['frame_by_selected_dim'] = ''

    # variances
    X = 100
    results['loss_std'] = np.std(results['training_data']['all_losses'][i_star-X:i_star+X])
    results['grad_beta_std'] = np.std(np.sqrt(results['training_data']['grads'])[i_star-X:i_star+X])
    results[f'loss_mean_around_i_star_{X}_steps'] = np.mean(results['training_data']['all_losses'][i_star-X:i_star+X])
    results[f'grad_beta_mean_around_i_star_{X}_steps'] = np.mean(np.sqrt(results['training_data']['grads'])[i_star-X:i_star+X])


    # get feture differences
    # Calculate cumulative mean and std dev up to i_star
    features_array = np.mean(results['training_data']['features_data'][:i_star+1], axis=1)  # Include i_star
    num_timesteps = features_array.shape[0]

    # # Calculate cumulative means at each timestep
    cumulative_means = np.zeros_like(features_array)
    for t in range(num_timesteps):
        cumulative_means[t] = np.mean(features_array[:t+1], axis=0)

    # wrt features model  
    features_model = np.array(results['training_data']['features_model'][:i_star+1])
    terminus_features_mean = features_model[-1].mean(axis=0)
    terminus_features_std = features_model[-1].std(axis=0)

    features_model_mean_last_100_steps = features_model[-X:].mean(axis=1).mean(axis=0)
    features_model_std_last_100_steps = features_model[-X:].mean(axis=1).std(axis=0)

    terminus_feature_diffs = np.abs(cumulative_means[-1] - terminus_features_mean)
    terminus_feature_diffs_std = np.std(cumulative_means[-1] - features_model[-1], axis=0)
    max_idx_feature_diffs = np.argmax(terminus_feature_diffs)

    if convert_np_arrays_to_list:
        features_data_running_mean = cumulative_means[-1].tolist()
        features_data_running_mean_std = cumulative_means[-X:].std(axis=0).tolist()
        terminus_features_mean = terminus_features_mean.tolist()
        terminus_features_std = terminus_features_std.tolist()
        features_model_mean_last_100_steps = features_model_mean_last_100_steps.tolist()
        features_model_std_last_100_steps = features_model_std_last_100_steps.tolist()
        terminus_feature_diffs = terminus_feature_diffs.tolist()
        terminus_feature_diffs_std = terminus_feature_diffs_std.tolist()
    else:
        features_data_running_mean = cumulative_means[-1]
        features_data_running_mean_std = cumulative_means[-X:].std(axis=0)
        terminus_features_mean = terminus_features_mean
        terminus_features_std = terminus_features_std
        features_model_mean_last_100_steps = features_model_mean_last_100_steps
        features_model_std_last_100_steps = features_model_std_last_100_steps
        terminus_feature_diffs = terminus_feature_diffs
        terminus_feature_diffs_std = terminus_feature_diffs_std

    results['features_data_running_mean'] = features_data_running_mean
    results['features_data_running_mean_std'] = features_data_running_mean_std
    results['terminus_features_mean'] = terminus_features_mean
    results['terminus_features_std'] = terminus_features_std
    results['features_model_mean_last_100_steps'] = features_model_mean_last_100_steps
    results['features_model_std_last_100_steps'] = features_model_std_last_100_steps
    results['max_idx_feature_diffs'] = max_idx_feature_diffs
    results['terminus_feature_diffs'] = terminus_feature_diffs
    results['terminus_feature_diffs_std'] = terminus_feature_diffs_std

    # results['betas'] = results['training_data']['betas']
    # results['o_mappings'] = results['training_data']['o_mappings']
    # results['losses'] = results['training_data']['all_losses']
    # results['best_betas'] = results['training_data']['best_betas']
    # results['betas'] = results['training_data']['betas']
    # results['features_data'] = results['training_data']['features_data']
    # results['features_model'] = results['training_data']['features_model']
    # results['attn_weights'] = results['training_data']['attn_weights']
    # results['best_selected_dim'] = results['training_data']['best_selected_dim']


    results.pop('training_data')
    for key in IGNORE_KEYS:
        results.pop(key)

    # Convert any numpy int64/float64 types to native Python types for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_to_serializable(x) for x in obj]
        elif isinstance(obj, np.ndarray):
            return convert_to_serializable(obj.tolist())
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        return obj

    return convert_to_serializable(results)
