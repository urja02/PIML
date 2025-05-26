import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os




with open('/content/drive/MyDrive/GAT_LET_Analysis/training/ZS_test', 'rb') as fp:
    ZS_new = pickle.load(fp)


with open('/content/drive/MyDrive/GAT_LET_Analysis/training/xs', 'rb') as fp:
    xs = pickle.load(fp)

with open('/content/drive/MyDrive/GAT_LET_Analysis/evaluation/batched_graph_test.pkl', 'rb') as fp:
    batched_graph_test = pickle.load(fp)


xs_converted= xs*2.54
ZS_converted = [i*2.54 for i in ZS_new]
xs=np.round(xs_converted,2)
ZS_new=[np.round(zs,2) for zs in ZS_converted]

pred_graph ={}
current_index = 0
for i in range(batched_graph_test.batch_size):
    res_test= len(batched_graph_test[i].y)

    pred_values = final_y_pred[current_index:current_index+res_test]

    pred_graph[i]=pred_values
    current_index+=res_test


sns.set_style("darkgrid")

stress_z = final_y_plot[:, 0]

plt.scatter(stress_z, final_y_pred[:, 0])
p1= max(max(stress_z), max(stress_z))
p2=min(min(stress_z), min(stress_z))
plt.plot([p2,p1],[p2,p1],color='black', linestyle='--')
plt.xlabel('Actual $\mu\epsilon_z$', fontsize=12)
plt.ylabel('Predicted $\mu\epsilon_z$', fontsize=12)
plt.title('Actual vs Predicted in $\mu\epsilon_z$', fontsize=12)

# Set tick parameters with font size 12 pt
plt.tick_params(axis='both', which='major', labelsize=12)
plt.show()
plt.close()


stress_r =  final_y_plot[:, 1]
plt.scatter(stress_r, final_y_pred[:, 1])
p1= max(max(stress_r), max(stress_r))
p2=min(min(stress_r), min(stress_r))
plt.plot([p2,p1],[p2,p1],color='black', linestyle='--')
plt.xlabel('Actual $\mu\epsilon_r$', fontsize=12)
plt.ylabel('Predicted $\mu\epsilon_r$', fontsize=12)
plt.title('Actual vs Predicted in $\mu\epsilon_r$', fontsize=12)

# Set tick parameters with font size 12 pt
plt.tick_params(axis='both', which='major', labelsize=12)
plt.show()
plt.close()

stress_t =final_y_plot[:, 2]
plt.scatter(stress_t,  final_y_pred[:, 2])
p1= max(max(stress_t), max(stress_t))
p2=min(min(stress_t), min(stress_t))
plt.plot([p2,p1],[p2,p1],color='black', linestyle='--')
plt.xlabel('Actual $\mu\epsilon_t$', fontsize=12)
plt.ylabel('Predicted $\mu\epsilon_t$', fontsize=12)
plt.title('Actual vs Predicted in $\mu\epsilon_t$', fontsize=12)

# Set tick parameters with font size 12 pt
plt.tick_params(axis='both', which='major', labelsize=12)
plt.show()
plt.close()


test_struct=0
test_g_struct=0



sns.set(rc={'figure.figsize':(30,20)},font_scale=3)
z=np.array(ZS_new[test_struct])
response='Strain_Z'
A_prep=np.reshape(batched_graph_test[test_g_struct].y[:,0],(len(ZS_new[test_struct]),len(xs)))
heatmap = sns.heatmap(A_prep,linewidths=.5,xticklabels=xs,yticklabels=-z,   # Annotation font size
            cbar_kws={'label': 'Strain (µε)'})  # Colorbar label font size
colorbar = heatmap.collections[0].colorbar
colorbar.ax.yaxis.label.set_size(30)
plt.xlabel('x (cm)', fontsize=30)  # X-axis label font size
plt.ylabel('z (cm)', fontsize=30)  # Y-axis label font size
plt.title(response)
plt.xticks(fontsize=30)  # X-axis ticks font size
plt.yticks(fontsize=30)
plt.tight_layout()
plt.show()
plt.close()

sns.set(rc={'figure.figsize':(20,10)},font_scale=3)
z=np.array(ZS_new[test_struct])
response='Strain_Z'
A_bar=np.reshape(pred_graph[test_struct][:,0],(len(ZS_new[test_struct]),len(xs)))
heatmap = sns.heatmap(A_bar, linewidths=.5,xticklabels=xs,yticklabels=-z,   # Annotation font size
            cbar_kws={'label': 'Strain (µε)'})  # Colorbar label font size
colorbar = heatmap.collections[0].colorbar
colorbar.ax.yaxis.label.set_size(30)
plt.xlabel('x (cm)', fontsize=30)  # X-axis label font size
plt.ylabel('z (cm)', fontsize=30)  # Y-axis label font size
plt.title(response)
plt.xticks(fontsize=30)  # X-axis ticks font size
plt.yticks(fontsize=30)
plt.title('$\epsilon_z$')
plt.tight_layout()

plt.savefig(os.path.join(plot_dir, f"NN_strain_z_heatmap_struct0_pred.png"))
plt.close()
