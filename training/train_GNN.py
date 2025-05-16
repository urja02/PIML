
import time
from sklearn.metrics import mean_squared_error,mean_absolute_error,mean_absolute_percentage_error
import torch
from utils import setup_logging
import seaborn as sns
import torch.nn as nn


def training_loop(args, model, optimizer, batched_graph,batched_graph_val, batched_graph_test, lr, epochs,criterion, Section, ZS,xs,weighted_loss=None):

  weight_dir = setup_logging(args)

 
  device  = torch.device('cuda')
  model = model.to(device)
  

  train_losses=[]
  val_losses=[]
  train_error_z = []
  train_error_r = []
  train_error_t = []


  val_error_z = []
  val_error_r = [] 
  val_error_t = [] 
  def train(data):
    data=data.cuda()

    model.train()
    optimizer.zero_grad()  # Clear gradients.
    out = model(data.x,data.edge_index)
    
    loss = criterion(out,data.y)
     
      
    loss.backward()
    optimizer.step()  # Update parameters based on gradients.
    
    train_mse_stress_z = mean_squared_error(data.y[:,0].cpu().detach().numpy(), out[:, 0].cpu().detach().numpy())
    # train_mse_stress_z = np.sqrt(train_mse_stress_z )
    train_mse_stress_r = mean_squared_error(data.y[:,1].cpu().detach().numpy(), out[:, 1].cpu().detach().numpy())
    # train_mse_stress_r = np.sqrt(train_mse_stress_r)
    train_mse_stress_t = mean_squared_error(data.y[:,2].cpu().detach().numpy(), out[:, 2].cpu().detach().numpy())
    # train_mse_stress_t = np.sqrt(train_mse_stress_t)



    return loss,train_mse_stress_z,train_mse_stress_r,train_mse_stress_t,out

  def validate(data):
    data=data.cuda()
    model.eval()
   
    with torch.no_grad():  # Disable gradient computation during validation
      out = model(data.x, data.edge_index)  # Perform forward pass

      val_loss = criterion(out, data.y)
        # Compute loss on validation set
      val_mse_stress_z = mean_squared_error(data.y[:,0].cpu().detach().numpy(), out[:, 0].cpu().detach().numpy())
      # val_mse_stress_z = np.sqrt(val_mse_stress_z)
      val_mse_stress_r =mean_squared_error(data.y[:,1].cpu().detach().numpy(), out[:, 1].cpu().detach().numpy())
      # val_mse_stress_r = np.sqrt(val_mse_stress_r)
      val_mse_stress_t =mean_squared_error(data.y[:,2].cpu().detach().numpy(), out[:, 2].cpu().detach().numpy())
      # val_mse_stress_t = np.sqrt(val_mse_stress_t)

      return val_loss, val_mse_stress_z,val_mse_stress_r,val_mse_stress_t,out




