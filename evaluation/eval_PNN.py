import torch
from PIML.model import NeuralNetwork
from sklearn.preprocessing import StandardScaler
import numpy as np
from .utils import setup_plotting, load_pickle, convert_units, build_pred_graph
import logging
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_actual_vs_predicted_pnn(target_test, predicted_values, plot_dir=None):
    sns.set_style("darkgrid")
    labels = ['z', 'r', 't']
    for i, label in enumerate(labels):
        plt.scatter(target_test[:, i], predicted_values[:, i])
        min_val = min(target_test[:, i].min(), predicted_values[:, i].min())
        max_val = max(target_test[:, i].max(), predicted_values[:, i].max())
        plt.plot([min_val, max_val], [min_val, max_val], color='black', linestyle='--')
        plt.xlabel(f'Actual $\mu\epsilon_{label}$', fontsize=12)
        plt.ylabel(f'Predicted $\mu\epsilon_{label}$', fontsize=12)
        plt.title(f'Actual vs Predicted in $\mu\epsilon_{label}$', fontsize=12)
        plt.tick_params(axis='both', which='major', labelsize=12)
        plt.xlim(min_val, max_val)
        plt.ylim(min_val, max_val)
        plt.tight_layout()
        if plot_dir:
            plt.savefig(os.path.join(plot_dir, f"actual_vs_pred_{label}.png"))
        plt.close()

def plot_heatmaps(ZS_new, xs, batched_graph_test, pred_graph, plot_dir=None, test_struct=0, test_g_struct=0):
    # Strain_Z actual
    sns.set_theme(rc={'figure.figsize':(30,20)}, font_scale=3)
    z = np.array(ZS_new[test_struct])
    response = 'Strain_Z'
    A_prep = np.reshape(batched_graph_test[test_g_struct].y[:,0], (len(ZS_new[test_struct]), len(xs)))
    heatmap = sns.heatmap(A_prep, linewidths=.5, xticklabels=xs, yticklabels=-z, cbar_kws={'label': 'Strain (µε)'})
    colorbar = heatmap.collections[0].colorbar
    colorbar.ax.yaxis.label.set_size(30)
    plt.xlabel('x (cm)', fontsize=30)
    plt.ylabel('z (cm)', fontsize=30)
    plt.title(response)
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    plt.tight_layout()
    if plot_dir:
        plt.savefig(os.path.join(plot_dir, "NN_strain_z_heatmap_struct0_actual.png"))
    plt.close()

    # Strain_Z predicted
    sns.set_theme(rc={'figure.figsize':(20,10)}, font_scale=3)
    A_bar = np.reshape(pred_graph[test_struct][:,0], (len(ZS_new[test_struct]), len(xs)))
    heatmap = sns.heatmap(A_bar, linewidths=.5, xticklabels=xs, yticklabels=-z, cbar_kws={'label': 'Strain (µε)'})
    colorbar = heatmap.collections[0].colorbar
    colorbar.ax.yaxis.label.set_size(30)
    plt.xlabel('x (cm)', fontsize=30)
    plt.ylabel('z (cm)', fontsize=30)
    plt.title('$\epsilon_z$')
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    plt.tight_layout()
    if plot_dir:
        plt.savefig(os.path.join(plot_dir, "NN_strain_z_heatmap_struct0_pred.png"))
    plt.close()

    # Strain_R actual
    sns.set_theme(rc={'figure.figsize':(30,20)}, font_scale=3)
    response = 'Strain_R'
    A_prep = np.reshape(batched_graph_test[test_g_struct].y[:,1], (len(ZS_new[test_struct]), len(xs)))
    heatmap = sns.heatmap(A_prep, linewidths=.5, xticklabels=xs, yticklabels=-z, cbar_kws={'label': 'Strain (µe)'})
    colorbar = heatmap.collections[0].colorbar
    colorbar.ax.yaxis.label.set_size(30)
    plt.xlabel('x (cm)', fontsize=30)
    plt.ylabel('z (cm)', fontsize=30)
    plt.title(response)
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    plt.tight_layout()
    if plot_dir:
        plt.savefig(os.path.join(plot_dir, "NN_strain_r_heatmap_struct0_actual.png"))
    plt.close()

    # Strain_R predicted
    sns.set_theme(rc={'figure.figsize':(20,10)}, font_scale=3)
    A_bar = np.reshape(pred_graph[test_struct][:,1], (len(ZS_new[test_struct]), len(xs)))
    heatmap = sns.heatmap(A_bar, linewidths=.5, xticklabels=xs, yticklabels=-z, cbar_kws={'label': 'Strain (µe)'})
    colorbar = heatmap.collections[0].colorbar
    colorbar.ax.yaxis.label.set_size(30)
    plt.xlabel('x (cm)', fontsize=30)
    plt.ylabel('z (cm)', fontsize=30)
    plt.title('$\epsilon_r$')
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    plt.tight_layout()
    if plot_dir:
        plt.savefig(os.path.join(plot_dir, "NN_strain_r_heatmap_struct0_pred.png"))
    plt.close()

def evaluate_PNN(args, TEST, TEST_out, ZS_path, xs_path, batched_graph_test_path):
    torch.manual_seed(42)
    model = NeuralNetwork()
    model.load_state_dict(torch.load(args.model_path))
    model.eval()
    
    with torch.no_grad():
        test_inputs = torch.tensor(np.vstack(TEST),dtype=torch.float32)
        predicted_values = model(test_inputs)
        target_test = np.vstack(TEST_out)
    
    plot_dir = setup_plotting()
    
    mse_z = mean_squared_error( target_test[:, 0], predicted_values[:, 0])
    mae_z = mean_absolute_error( target_test[:, 0], predicted_values[:, 0])
    mse_r = mean_squared_error( target_test[:, 1], predicted_values[:, 1])
    mae_r = mean_absolute_error(target_test[:, 1], predicted_values[:, 1])
    mse_t = mean_squared_error( target_test[:, 2], predicted_values[:, 2])
    mae_t = mean_absolute_error(target_test[:, 2], predicted_values[:, 2])

    logging.info(f"MSE_Z: {mse_z}, MAE_Z: {mae_z}, MSE_R: {mse_r}, MAE_R: {mae_r}, MSE_T: {mse_t}, MAE_T: {mae_t}")

    # Load data
    ZS_new = load_pickle(ZS_path)
    xs = load_pickle(xs_path)
    batched_graph_test = load_pickle(batched_graph_test_path)
    
    # Convert units
    xs, ZS_new = convert_units(xs, ZS_new)

    # Build prediction graph
    pred_graph = build_pred_graph(batched_graph_test, predicted_values)
    
    # Generate and save actual vs predicted plots
    plot_actual_vs_predicted_pnn(target_test, predicted_values, plot_dir=plot_dir)

    # Plot heatmaps
    plot_heatmaps(ZS_new, xs, batched_graph_test, pred_graph, plot_dir=plot_dir)

