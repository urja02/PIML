from .utils import setup_plotting, load_pickle, convert_units, build_pred_graph
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_heatmaps(ZS_new, xs, batched_graph_test, pred_graph, plot_dir=None, test_struct=0, test_g_struct=0):
    # Strain_Z actual
    sns.set_theme(rc={'figure.figsize':(20,10)}, font_scale=3)
    z = np.array(ZS_new[test_struct])
    response = 'Strain_Z'
    A_prep = np.reshape(batched_graph_test[test_g_struct].y[:,0], (len(ZS_new[test_struct]), len(xs)))
    heatmap = sns.heatmap(A_prep, linewidths=.5, xticklabels=xs, yticklabels=-z, cbar_kws={'label': 'Strain (µε)'})
    colorbar = heatmap.collections[0].colorbar
    colorbar.ax.yaxis.label.set_size(30)
    plt.xlabel('x (cm)', fontsize=30)
    plt.ylabel('z (cm)', fontsize=30)
    plt.title(r'$\\epsilon_z$')
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
    plt.title(r'$\\epsilon_z$')
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    plt.tight_layout()
    if plot_dir:
        plt.savefig(os.path.join(plot_dir, "NN_strain_z_heatmap_struct0_pred.png"))
    plt.close()

    # Strain_R actual
    sns.set_theme(rc={'figure.figsize':(20,10)}, font_scale=3)
    response = 'Strain_R'
    A_prep = np.reshape(batched_graph_test[test_g_struct].y[:,1], (len(ZS_new[test_struct]), len(xs)))
    heatmap = sns.heatmap(A_prep, linewidths=.5, xticklabels=xs, yticklabels=-z, cbar_kws={'label': 'Strain (µε)'})
    colorbar = heatmap.collections[0].colorbar
    colorbar.ax.yaxis.label.set_size(30)
    plt.xlabel('x (cm)', fontsize=30)
    plt.ylabel('z (cm)', fontsize=30)
    plt.title(r'$\\epsilon_r$')
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    plt.tight_layout()
    if plot_dir:
        plt.savefig(os.path.join(plot_dir, "NN_strain_r_heatmap_struct0_actual.png"))
    plt.close()

    # Strain_R predicted
    sns.set_theme(rc={'figure.figsize':(20,10)}, font_scale=3)
    A_bar = np.reshape(pred_graph[test_struct][:,1], (len(ZS_new[test_struct]), len(xs)))
    heatmap = sns.heatmap(A_bar, linewidths=.5, xticklabels=xs, yticklabels=-z, cbar_kws={'label': 'Strain (µε)'})
    colorbar = heatmap.collections[0].colorbar
    colorbar.ax.yaxis.label.set_size(30)
    plt.xlabel('x (cm)', fontsize=30)
    plt.ylabel('z (cm)', fontsize=30)
    plt.title(r'$\\epsilon_r$')
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    plt.tight_layout()
    if plot_dir:
        plt.savefig(os.path.join(plot_dir, "NN_strain_r_heatmap_struct0_pred.png"))
    plt.close()


def plot_actual_vs_predicted_GNN(batched_graph_test, predicted_values, plot_dir=None):
    sns.set_style("darkgrid")
    labels = ['z', 'r', 't']
    for i, label in enumerate(labels):
        plt.scatter(batched_graph_test.y[:, i].cpu().detach().numpy(), predicted_values[:, i])
        p1 = max(batched_graph_test.y[:, i].cpu().detach().numpy().max(), predicted_values[:, i].max())
        p2 = min(batched_graph_test.y[:, i].cpu().detach().numpy().min(), predicted_values[:, i].min())
        plt.plot([p2, p1], [p2, p1], color='black', linestyle='--')
        plt.xlabel(f'Actual $\\mu\\epsilon_{label}$', fontsize=12)
        plt.ylabel(f'Predicted $\\mu\\epsilon_{label}$', fontsize=12)
        plt.title(f'Actual vs Predicted in $\\mu\\epsilon_{label}$', fontsize=12)
        plt.tight_layout()
        if plot_dir:
            plt.savefig(os.path.join(plot_dir, f"actual_vs_pred_{label}.png"))
        plt.close()

def evaluate_GNN(args, batched_graph_test_path, model, ZS_path, xs_path):
    plot_dir = setup_plotting()
    batched_graph_test = load_pickle(batched_graph_test_path)
    
    model.load_state_dict(torch.load(args.model_path))

    model.eval()

    with torch.no_grad():
        test_outputs = model(batched_graph_test.x,batched_graph_test.edge_index)
        predicted_values = test_outputs.numpy()

    mse_z = mean_squared_error( batched_graph_test.y[:, 0].cpu().detach().numpy(), predicted_values[:, 0])
    mse_r = mean_squared_error( batched_graph_test.y[:, 1].cpu().detach().numpy(), predicted_values[:, 1])
    mse_t = mean_squared_error( batched_graph_test.y[:, 2].cpu().detach().numpy(), predicted_values[:, 2])
    mae_z = mean_absolute_error( batched_graph_test.y[:, 0].cpu().detach().numpy(), predicted_values[:, 0])
    mae_r = mean_absolute_error(batched_graph_test.y[:, 1].cpu().detach().numpy(), predicted_values[:, 1])
    mae_t = mean_absolute_error(batched_graph_test.y[:, 2].cpu().detach().numpy(), predicted_values[:, 2])

    mape_z = mean_absolute_percentage_error(batched_graph_test.y[:, 0].cpu().detach().numpy(), predicted_values[:, 0])
    mape_r = mean_absolute_percentage_error(batched_graph_test.y[:, 1].cpu().detach().numpy(), predicted_values[:, 1])
    mape_t = mean_absolute_percentage_error(batched_graph_test.y[:, 2].cpu().detach().numpy(), predicted_values[:, 2])

    logging.info(f"MSE_Z: {mse_z}, MAE_Z: {mae_z}, MSE_R: {mse_r}, MAE_R: {mae_r}, MSE_T: {mse_t}, MAE_T: {mae_t}, MAPE_Z: {mape_z}, MAPE_R: {mape_r}, MAPE_T: {mape_t}")

    plot_actual_vs_predicted_GNN(batched_graph_test, predicted_values, plot_dir=plot_dir)

    # Load data
    ZS_new = load_pickle(ZS_path)
    xs = load_pickle(xs_path)
    batched_graph_test = load_pickle(batched_graph_test_path)

    # Convert units
    xs, ZS_new = convert_units(xs, ZS_new)

    # Build prediction graph
    pred_graph = build_pred_graph(batched_graph_test, predicted_values)

    # Plot heatmaps
    plot_heatmaps(ZS_new, xs, batched_graph_test, pred_graph, plot_dir=plot_dir)    


