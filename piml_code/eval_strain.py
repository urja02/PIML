import json
import time
from sklearn.metrics import mean_squared_error,mean_absolute_percentage_error,mean_absolute_error
import torch
import matplotlib.pyplot as plt
import numpy as np
from utils import setup_logging
import logging
import os
import seaborn as sns
import torch.nn as nn
import copy




def evaluation_strain(args,model,batched_graph,mins_train,maxs_train,Section,xs,ZS,batched_graph_test,test_struct,test_g_struct):
  
  print("printing the lengths of ZS")
  Section_copy = copy.deepcopy(Section)
 
  ZS_new=ZS
  print("M")
  print(batched_graph_test[0].y)
    
  



  _,plot_dir, pred_dir = setup_logging(args)

  path = '/work/pi_eokte_umass_edu/Urja/NEW_MODEL/z_strain_dropped_1500/6_neighbour_with_linear/denorm/0/checkpoints/model_epoch_14400.pt'
  checkpoint = torch.load(path)
  print(checkpoint.keys())
  # Extract the model state dictionary
  model_state_dict = checkpoint['model_state_dict']

  # Load the state dictionary into the model
  model.load_state_dict(model_state_dict)  
  model.eval()  # Set the model to evaluation mode
  with torch.no_grad():
    batched_graph_test = batched_graph_test.cpu()
    model=model.cpu()
    test_outputs = model(batched_graph_test.x,batched_graph_test.edge_index)

  with torch.no_grad():
    batched_graph = batched_graph.cpu()
    model=model.cpu()
    train_outputs = model(batched_graph.x,batched_graph.edge_index)


  # predicted_train = denormalize_min_max(train_outputs,mins_train,maxs_train )
  predicted_train = train_outputs.numpy()
  batched_graph.y = batched_graph.y.cpu().detach().numpy()
  Strain_z = batched_graph.y[:,0]
  plt.scatter(Strain_z, predicted_train[:,0])
  # for i in range(len(Strain_z)):
  #   plt.plot([predicted_values[i,0], predicted_values[i,0]], [Strain_z[i], Strain_z[i] - residuals[i]], color='green', linestyle='--', linewidth=1)
  plt.plot([np.min(Strain_z),np.max(Strain_z)],[np.min(Strain_z),np.max(Strain_z)],color='black', linestyle='--')
  plt.xlabel('Actual')
  # plt.xlim([0,200])
  # plt.ylim([0,200])
  plt.ylabel('Predicted')
  plt.title('Actual vs Predicted in z')
  plt.savefig(os.path.join(plot_dir, f"actual_vs_predicted_z_train_{args.lr}.png"))
  plt.close() 

  # Convert predictions to numpy array
  predicted_values = test_outputs.numpy()
  # predicted_values = denormalize_min_max(predicted_values,mins_train,maxs_train )



  batched_graph_test.y = batched_graph_test.y
  
  # true_test_strain_R=batched_graph_test[test_g_struct].y[:,2]
  # plt.hist(true_test_strain_R,bins=90)
  # plt.savefig(os.path.join(plot_dir,f"histogram of training input R{args.lr}.png"))
  # plt.close()
  mse_z = mean_squared_error( batched_graph_test.y[:, 0].cpu().detach().numpy(), predicted_values[:, 0])
  mse_r = mean_squared_error( batched_graph_test.y[:, 1].cpu().detach().numpy(), predicted_values[:, 1])
  mse_t = mean_squared_error( batched_graph_test.y[:, 2].cpu().detach().numpy(), predicted_values[:, 2])
  mae_z = mean_absolute_error( batched_graph_test.y[:, 0].cpu().detach().numpy(), predicted_values[:, 0])
  mae_r = mean_absolute_error(batched_graph_test.y[:, 1].cpu().detach().numpy(), predicted_values[:, 1])
  mae_t = mean_absolute_error(batched_graph_test.y[:, 2].cpu().detach().numpy(), predicted_values[:, 2])

  mape_z = mean_absolute_percentage_error(batched_graph_test.y[:, 0].cpu().detach().numpy(), predicted_values[:, 0])
  mape_r = mean_absolute_percentage_error(batched_graph_test.y[:, 1].cpu().detach().numpy(), predicted_values[:, 1])
  mape_t = mean_absolute_percentage_error(batched_graph_test.y[:, 2].cpu().detach().numpy(), predicted_values[:, 2])
  plt.hist((batched_graph_test.y[:, 1].cpu().detach().numpy()-predicted_values[:, 1])**2,40)
  mse_man=(np.mean(np.abs((batched_graph_test.y[:, 1].cpu().detach().numpy()-predicted_values[:, 1]))))
  logging.info(f'Max {np.max(batched_graph_test.y[:, 1].cpu().detach().numpy()-predicted_values[:, 1])}')
  plt.title(mse_man)
  plt.savefig(os.path.join(plot_dir, f"histogram_training{args.lr}.png"))
  plt.close() 


  logging.info(f'mean_squared_error (MSE_z): {mse_z:.4f}')
  logging.info(f'mean_squared_error (MSE_r): {mse_r:.4f}')
  logging.info(f'mean_squared_error (MSE_t): {mse_t:.4f}')

  logging.info(f'mean_abs_error (MSE_z): {mae_z:.4f}')
  logging.info(f'mean_abs_error (MSE_r): {mae_r:.4f}')
  logging.info(f'mean_abs_error (MSE_t): {mae_t:.4f}')

  logging.info(f'mean_abs_percentage_error (MSE_z): {mae_z:.4f}')
  logging.info(f'mean_abs_percentage_error (MSE_r): {mae_r:.4f}')
  logging.info(f'mean_abs_percentage_error (MSE_t): {mae_t:.4f}')
  # logging.info(f"information of the first test structure,{Section[901]}")
#   print( type(Section[test_struct]['Response'][0]['Strain_Z']))

#   print(batched_graph_test.x)
  res_test= [i for i in batched_graph_test[test_g_struct].y.cpu().detach().numpy()]
  print("printing res_tes")

  print(len(res_test))
  print("predicting")
  print(len(predicted_values))
  print(predicted_values)
  
  for i, j in zip(res_test, predicted_values[:len(res_test)]):
    logging.info(" ".join(["{:.2f}".format(i_) for i_ in i.tolist()]))
    logging.info(" ".join(["{:.2f}".format(i_) for i_ in j.tolist()]))
    logging.info("="*10)

  def convert_to_dict(sect):
    for k,v in sect.items():
      if isinstance(v,np.ndarray):
        sect[k]=v.tolist()
      elif isinstance(v,dict):
        convert_to_dict(v)
  section_str = copy.deepcopy(Section[test_struct])
  convert_to_dict(section_str)
    
    # Convert the dictionary to a JSON string
  section_str = json.dumps(section_str, indent=2)
    
    # Log the string
  logging.info(f'Information of the first test structure: {section_str}')


  Strain_z = batched_graph_test.y[:,0].cpu().detach().numpy()
  plt.scatter(Strain_z, predicted_values[:,0])
  # for i in range(len(Strain_z)):
  #   plt.plot([predicted_values[i,0], predicted_values[i,0]], [Strain_z[i], Strain_z[i] - residuals[i]], color='green', linestyle='--', linewidth=1)
  plt.plot([0,np.max(Strain_z)],[0,np.max(Strain_z)],color='black', linestyle='--')
  plt.xlabel('Actual')
  # plt.xlim([0,200])
  # plt.ylim([0,200])
  plt.ylabel('Predicted')
  plt.title('Actual vs Predicted in z')
  plt.savefig(os.path.join(plot_dir, f"actual_vs_predicted_z_lr_{args.lr}.png"))
  plt.close() 


  Strain_z = batched_graph_test.y[:,0].cpu().detach().numpy()
  plt.scatter(Strain_z, predicted_values[:,0])
  # for i in range(len(Strain_z)):
  #   plt.plot([predicted_values[i,0], predicted_values[i,0]], [Strain_z[i], Strain_z[i] - residuals[i]], color='green', linestyle='--', linewidth=1)
  plt.plot([0,np.max(Strain_z)],[0,np.max(Strain_z)],color='black', linestyle='--')
  plt.xlabel('Actual')
  plt.xlim([0,200])
  plt.ylim([0,200])
  plt.ylabel('Predicted')
  plt.title('Actual vs Predicted in z')
  plt.savefig(os.path.join(plot_dir, f"actual_vs_predicted_z_lr_{args.lr}.png"))
  plt.close() 


  # Strain_z = batched_graph.y[:,0].cpu().detach().numpy()
  # plt.scatter(Strain_z, predicted_values[:,0])
  # # for i in range(len(Strain_z)):
  # #   plt.plot([predicted_values[i,0], predicted_values[i,0]], [Strain_z[i], Strain_z[i] - residuals[i]], color='green', linestyle='--', linewidth=1)
  # plt.plot([0,np.max(Strain_z)],[0,np.max( predicted_values[:,0])],color='black', linestyle='--')
  # plt.xlabel('Actual')
  # plt.xlim([0,200])
  # plt.ylim([0,200])
  # plt.ylabel('Predicted')
  # plt.title('Actual vs Predicted in z')
  # plt.savefig(os.path.join(plot_dir, f"actual_vs_predicted_z_lr_{args.lr}.png"))
  # plt.close() 
  residuals = batched_graph_test.y[:,0].cpu().detach().numpy() - predicted_values[:,0]
    # Calculate residuals

  # Plot Residuals vs Actual Values
  plt.scatter(batched_graph_test.y[:,0].cpu().detach().numpy(), residuals)
  plt.axhline(y=0, color='black', linestyle='--')
  plt.xlabel('Actual Values')
  plt.ylabel('Residuals')
  plt.title('Residuals vs Actual Values')
  plt.savefig(os.path.join(plot_dir, f"residuals_vs_actuals_z{args.lr}.png"))
  plt.close()

  # Plot Residuals vs Predicted Values
  plt.scatter( predicted_values[:,0], residuals)
  plt.axhline(y=0, color='black', linestyle='--')
  plt.xlabel('Predicted Values')
  plt.ylabel('Residuals')
  plt.title('Residuals vs Predicted Values')
  plt.savefig(os.path.join(plot_dir, f"residuals_vs_predicted_z{args.lr}.png"))
  plt.close()

  Strain_r = batched_graph_test.y[:,1].cpu().detach().numpy()
  print("Strain_r")
  print(len(Strain_r))
  plt.scatter(Strain_r, predicted_values[:,1])
  plt.plot([np.min(Strain_r),np.max(Strain_r)],[np.min(Strain_r),np.max( predicted_values[:,1])],color='black', linestyle='--')
  plt.xlabel('Actual')
  plt.ylabel('Predicted')
  plt.title('Actual vs Predicted in r')
  plt.savefig(os.path.join(plot_dir, f"actual_vs_predicted_r_lr_{args.lr}.png"))
  plt.close() 

    # Calculate residuals
  residuals = batched_graph_test.y[:,1].cpu().detach().numpy() - predicted_values[:,1]
  # Plot Residuals vs Actual Values
  plt.scatter(batched_graph_test.y[:,1].cpu().detach().numpy(), residuals)
  plt.axhline(y=0, color='black', linestyle='--')
  plt.xlabel('Actual Values')
  plt.ylabel('Residuals')
  plt.title('Residuals vs Actual Values')
  plt.savefig(os.path.join(plot_dir, f"residuals_vs_actuals_r{args.lr}.png"))
  plt.close()

  # Plot Residuals vs Predicted Values
  plt.scatter(predicted_values[:,1], residuals)
  plt.axhline(y=0, color='black', linestyle='--')
  plt.xlabel('Predicted Values')
  plt.ylabel('Residuals')
  plt.title('Residuals vs Predicted Values')
  plt.savefig(os.path.join(plot_dir, f"residuals_vs_predicted_r{args.lr}.png"))
  plt.close()

  Strain_t = batched_graph_test.y[:,2].cpu().detach().numpy()
  plt.scatter(Strain_t, predicted_values[:,2])
  plt.plot([np.min(Strain_t),np.max(Strain_t)],[np.min(Strain_t),np.max( predicted_values[:,2])],color='black', linestyle='--')
  plt.xlabel('Actual')
  plt.ylabel('Predicted')
  plt.title('Actual vs Predicted in t')
  plt.savefig(os.path.join(plot_dir, f"actual_vs_predicted_t_lr_{args.lr}.png"))
  plt.close() 

  # Calculate residuals
  residuals = batched_graph_test.y[:,2].cpu().detach().numpy() - predicted_values[:,2]

  # Plot Residuals vs Actual Values
  plt.scatter(batched_graph_test.y[:,2].cpu().detach().numpy(), residuals)
  plt.axhline(y=0, color='black', linestyle='--')
  plt.xlabel('Actual Values')
  plt.ylabel('Residuals')
  plt.title('Residuals vs Actual Values')
  plt.savefig(os.path.join(plot_dir, f"residuals_vs_actuals_t{args.lr}.png"))
  plt.close()

  # Plot Residuals vs Predicted Values
  plt.scatter( predicted_values[:,2], residuals)
  plt.axhline(y=0, color='black', linestyle='--')
  plt.xlabel('Predicted Values')
  plt.ylabel('Residuals')
  plt.title('Residuals vs Predicted Values')
  plt.savefig(os.path.join(plot_dir, f"residuals_vs_predicted_t{args.lr}.png"))
  plt.close()

  sns.set(rc={'figure.figsize':(20,10)},font_scale=1.15)
  z=np.array(ZS[test_struct])
  response='Strain_Z'
  A=Section[test_struct]['Response'][0][response]
  print("debugging")
  print(type(A))
  sns.heatmap(A,annot=True, fmt=".4f", linewidths=.5,xticklabels=xs,yticklabels=-z,cbar_kws={'label': 'Strain (psi)'})
  plt.xlabel('x')
  plt.ylabel('z')
  plt.title(response)
  plt.savefig(os.path.join(plot_dir, f"heatmap_actual_z{args.lr}.png"))
  plt.close() 
 
  
  sns.set(rc={'figure.figsize':(20,10)},font_scale=1.15)
  z=np.array(ZS_new[test_struct])
  response='Strain_Z'
  A_prep=np.reshape(batched_graph_test[test_g_struct].y[:,0],(len(ZS_new[test_struct]),len(xs)))
  sns.heatmap(A_prep,annot=True, fmt=".4f", linewidths=.5,xticklabels=xs,yticklabels=-z,cbar_kws={'label': 'Strain (psi)'})
  plt.xlabel('x')
  plt.ylabel('z')
  plt.title(response)
  plt.savefig(os.path.join(plot_dir, f"heatmap_actual_z_prep{args.lr}.png"))
  plt.close() 
  
  
  sns.set(rc={'figure.figsize':(20,10)},font_scale=1.15)
  z=np.array(ZS_new[test_struct])
  response='Strain_Z'
  A_bar=np.reshape(predicted_values[:len(res_test),0],(len(ZS_new[test_struct]),len(xs)))
  sns.heatmap(A_bar,annot=True, fmt=".4f", linewidths=.5,xticklabels=xs,yticklabels=-z,cbar_kws={'label': 'Strain (psi)'})
  plt.xlabel('x')
  plt.ylabel('z')
  plt.title(response)
  plt.savefig(os.path.join(plot_dir, f"heatmap_predicted_z{args.lr}.png"))
  plt.close() 


  sns.set(rc={'figure.figsize':(20,10)},font_scale=1.15)
  z=np.array(ZS_new[test_struct])
  response='Strain_Z'
  A_sub=np.array(A_prep-A_bar)
  logging.info(f"Min difference of A_sub is {np.min(A_sub)} max difference is {np.max(A_sub)}")

  sns.heatmap(A_sub,annot=True, fmt=".4f", linewidths=.5,xticklabels=xs,yticklabels=-z,cbar_kws={'label': 'Strain (psi)'})
  plt.xlabel('x')
  plt.ylabel('z')
  plt.title(response)
  plt.savefig(os.path.join(plot_dir, f"heatmap_true_sub_z{args.lr}.png"))
  plt.close() 

  sns.set(rc={'figure.figsize':(20,10)},font_scale=1.15)
  z=np.array(ZS_new[test_struct])
  response='Strain_Z'
  A_sub=np.reshape(((A_prep-A_bar)/A_prep)*100,(len(ZS_new[test_struct]),len(xs)))
  sns.heatmap(A_sub,annot=True, fmt=".4f", linewidths=.5,xticklabels=xs,yticklabels=-z,cbar_kws={'label': 'Strain (psi)'})
  plt.xlabel('x')
  plt.ylabel('z')
  plt.title(response)
  plt.savefig(os.path.join(plot_dir, f"heatmap_true_sub_z_percentage{args.lr}.png"))
  plt.close() 


  sns.set(rc={'figure.figsize':(20,10)},font_scale=1.15)
  z=np.array(ZS_new[test_struct])
  response='Strain_R'
  A=Section[test_struct]['Response'][0][response]*1e6
  sns.heatmap(A,annot=True, fmt=".4f", linewidths=.5,xticklabels=xs,yticklabels=-z,cbar_kws={'label': 'Strain (psi)'})
  plt.xlabel('x')
  plt.ylabel('z')
  plt.title(response)
  plt.savefig(os.path.join(plot_dir, f"heatmap_actual_r{args.lr}.png"))
  plt.close() 

  sns.set(rc={'figure.figsize':(20,10)},font_scale=1.15)
  z=np.array(ZS_new[test_struct])
  response='Strain_R'
  A_prep=np.reshape(batched_graph_test[test_g_struct].y[:,1],(len(ZS_new[test_struct]),len(xs)))
  sns.heatmap(A_prep,annot=True, fmt=".4f", linewidths=.5,xticklabels=xs,yticklabels=-z,cbar_kws={'label': 'Strain (psi)'})
  plt.xlabel('x')
  plt.ylabel('z')
  plt.title(response)
  plt.savefig(os.path.join(plot_dir, f"heatmap_actual_r_prep{args.lr}.png"))
  plt.close() 



  sns.set(rc={'figure.figsize':(20,10)},font_scale=1.15)
  z=np.array(ZS_new[test_struct])
  response='Strain_R'
  A_bar=np.reshape(predicted_values[:len(res_test),1],(len(ZS_new[test_struct]),len(xs)))
  sns.heatmap(A_bar,annot=True, fmt=".4f", linewidths=.5,xticklabels=xs,yticklabels=-z,cbar_kws={'label': 'Strain (psi)'})
  plt.xlabel('x')
  plt.ylabel('z')
  plt.title(response)
  plt.savefig(os.path.join(plot_dir, f"heatmap_predicted_r{args.lr}.png"))
  plt.close() 

  sns.set(rc={'figure.figsize':(20,10)},font_scale=1.15)
  z=np.array(ZS_new[test_struct])
  response='Strain_R'
  A_sub=np.reshape(A_prep-A_bar,(len(ZS_new[test_struct]),len(xs)))
  sns.heatmap(A_sub,annot=True, fmt=".4f", linewidths=.5,xticklabels=xs,yticklabels=-z,cbar_kws={'label': 'Strain (psi)'})
  plt.xlabel('x')
  plt.ylabel('z')
  plt.title(response)
  plt.savefig(os.path.join(plot_dir, f"heatmap_true_sub_r{args.lr}.png"))
  plt.close() 

  sns.set(rc={'figure.figsize':(20,10)},font_scale=1.15)
  z=np.array(ZS_new[test_struct])
  response='Strain_R'
  A_sub=np.reshape(((A_prep-A_bar)/A_bar)*100,(len(ZS_new[test_struct]),len(xs)))
  sns.heatmap(A_sub,annot=True, fmt=".4f", linewidths=.5,xticklabels=xs,yticklabels=-z,cbar_kws={'label': 'Strain (psi)'})
  plt.xlabel('x')
  plt.ylabel('z')
  plt.title(response)
  plt.savefig(os.path.join(plot_dir, f"heatmap_true_sub_r{args.lr}.png"))
  plt.close() 
  
  sns.set(rc={'figure.figsize':(20,10)},font_scale=1.15)
  z=np.array(ZS_new[test_struct])
  response='Strain_T'
  A=Section[test_struct]['Response'][0][response]*1e6
  sns.heatmap(A,annot=True, fmt=".4f", linewidths=.5,xticklabels=xs,yticklabels=-z,cbar_kws={'label': 'Strain (psi)'})
  plt.xlabel('x')
  plt.ylabel('z')
  plt.title(response)
  plt.savefig(os.path.join(plot_dir, f"heatmap_actual_t{args.lr}.png"))
  plt.close() 

  sns.set(rc={'figure.figsize':(20,10)},font_scale=1.15)
  z=np.array(ZS_new[test_struct])
  response='Strain_T'
  A_prep=np.reshape(batched_graph_test[test_g_struct].y[:,2],(len(ZS_new[test_struct]),len(xs)))
  sns.heatmap(A_prep,annot=True, fmt=".4f", linewidths=.5,xticklabels=xs,yticklabels=-z,cbar_kws={'label': 'Strain (psi)'})
  plt.xlabel('x')
  plt.ylabel('z')
  plt.title(response)
  plt.savefig(os.path.join(plot_dir, f"heatmap_actual_t_prep{args.lr}.png"))
  plt.close() 



  sns.set(rc={'figure.figsize':(20,10)},font_scale=1.15)
  z=np.array(ZS_new[test_struct])
  response='Strain_T'
  A_pred=np.reshape(predicted_values[:len(res_test),2],(len(ZS_new[test_struct]),len(xs)))
  sns.heatmap(A_pred,annot=True, fmt=".4f", linewidths=.5,xticklabels=xs,yticklabels=-z,cbar_kws={'label': 'Strain (psi)'})
  plt.xlabel('x')
  plt.ylabel('z')
  plt.title(response)
  plt.savefig(os.path.join(plot_dir, f"heatmap_predicted_t{args.lr}.png"))
  plt.close() 

  sns.set(rc={'figure.figsize':(20,10)},font_scale=1.15)
  z=np.array(ZS_new[test_struct])
  response='Strain_T'
  A_sub=np.reshape(A_prep-A_pred,(len(ZS_new[test_struct]),len(xs))) # dont reshape
  sns.heatmap(A_sub,annot=True, fmt=".4f", linewidths=.5,xticklabels=xs,yticklabels=-z,cbar_kws={'label': 'Strain (psi)'})
  plt.xlabel('x')
  plt.ylabel('z')
  plt.title(response)
  plt.savefig(os.path.join(plot_dir, f"heatmap_true_sub_t{args.lr}.png"))
  plt.close() 

  sns.set(rc={'figure.figsize':(20,10)},font_scale=1.15)
  z=np.array(ZS_new[test_struct])
  response='Strain_T'
  A_sub=np.reshape(((A_prep-A_pred)/A_prep)*100,(len(ZS_new[test_struct]),len(xs)))
  sns.heatmap(A_sub,annot=True, fmt=".4f", linewidths=.5,xticklabels=xs,yticklabels=-z,cbar_kws={'label': 'Strain (psi)'})
  plt.xlabel('x')
  plt.ylabel('z')
  plt.title(response)
  plt.savefig(os.path.join(plot_dir, f"heatmap_true_sub1_t{args.lr}.png"))
  plt.close() 
  







