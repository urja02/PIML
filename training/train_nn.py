import os
import seaborn as sns
import torch.nn as nn
import time
from sklearn.metrics import mean_squared_error,mean_absolute_error,mean_absolute_percentage_error
import torch
import matplotlib.pyplot as plt
import numpy as np
from utils import setup_logging
import logging

def denormalize_min_max(normalized_data, min_val, max_val):
    print("denormalizing")
    print((normalized_data * (max_val - min_val)) + min_val)
    
    return (normalized_data * (max_val - min_val)) + min_val

def training_nn(args,inputs,targets,input_val,target_val,test,target_test, Section, ZS,xs,mins_train,maxs_train):
    torch.manual_seed(42)
    class NeuralNetwork(nn.Module):
        def __init__(self, output_size):
            super(NeuralNetwork, self).__init__()

            # self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
            # self.flatten = nn.Flatten()
            self.f1 = nn.Linear(input_size, hidden_size)
            self.relu = nn.ReLU()
            self.f2 = nn.Linear(hidden_size, hidden_size)
            self.f3 = nn.Linear(hidden_size, output_size)


        def forward(self, x):
            x = self.f1(x)
            x = self.relu(x)
            x = self.f2(x)
            x = self.relu(x)
            x= self.f2(x)
            x= self.relu(x)
            x=self.f2(x)
            x= self.relu(x)
            x=self.f2(x)
            x= self.relu(x)
            x = self.f3(x)

            return x

    # Define the input size, hidden size, and output size
    input_size = 5 # Number of input features
    hidden_size =  90 # Number of neurons in the hidden layer
    output_size = 3 # Number of output features

    # Create the model
    model = NeuralNetwork(output_size)
    weight_dir, plot_dir, pred_dir = setup_logging(args)
    if args.train_nn:
        model=model.cuda()

        inputs=torch.tensor(np.vstack(inputs),dtype=torch.float32)
        
        inputs=inputs.cuda()
        
        targets = torch.tensor(np.vstack(targets),dtype=torch.float32)
        targets = targets.cuda()
        train_losses=[]
        val_losses=[]
        
        criterion = nn.L1Loss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        num_epochs=15000
        input_val=torch.tensor(np.vstack(input_val), dtype=torch.float32).cuda()
        target_val = torch.tensor(np.vstack(target_val), dtype=torch.float32).cuda()

        # Train the model
        for epoch in range(num_epochs):
            # Forward pass
            model.train()
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.mean().item())
        
            model.eval()
            with torch.no_grad():
                outputs_val = model(input_val)
                loss_val = criterion(outputs_val, target_val)
                val_losses.append(loss_val.item())

            # Print progress
            if (epoch+1) % 100 == 0:
                logging.info(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {loss.item():.4f}, Val_loss: {loss_val.item():.4f}')
                torch.save(model.state_dict(), os.path.join(weight_dir, f"model_epoch_{epoch}.pt"))
            
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.xscale('log')
        # plt.xlim(left=1000)
        plt.title('Training and Validation Losses vs. Epochs')
        plt.savefig(os.path.join(plot_dir,f"train_vs_val_loss_lr_{args.lr}.png"))
        plt.close() 

    
        

