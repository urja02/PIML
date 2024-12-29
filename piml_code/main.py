import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import pickle
from dataset_generation import generatesection, plot_sample_query_points,generate_query_points,analysis
from data_preprocessing import frame_filtering,  filtering_ZS, train_val_test_generate, formation_of_matrices,remove_strain_z
from graphs_formation import train_graph, val_graph, test_graph
from training import training_loop
from model import GCN, GCNWithMLP, GAT
from eval import evaluation
from eval_strain import evaluation_strain
import logging
from train_nn import training_nn

def parse_arguments():
    parser = argparse.ArgumentParser(description="Description of your program")
    
    parser.add_argument("--run_analysis", action="store_true", default=False, help="Set this flag to run analysis")
    parser.add_argument("--mode", type=str,choices=["train","eval"], help="either train mode or eval mode")
    parser.add_argument("--predict", type=str,choices=["predict_strain"], help="either stress or strain")
    parser.add_argument("--nn", action="store_true", default=False, help="either stress or strain")

    parser.add_argument("--train_nn", action="store_true", default=False, help="either stress or strain")
    parser.add_argument("--eval_nn",action="store_true", default=False, help="either stress or strain")

    # Add more arguments as needed
    parser.add_argument("--frame_large_path", type=str, help="Path to FrameLarge pickle file.")
    parser.add_argument("--section_path", type=str, help="Path to Section pickle file.")
    
    parser.add_argument("--model_type", type=str, default="GCN", choices=["GCN", "GCNWithMLP","GAT"],
                        help="Type of model to train (GCN or GCNWithMLP). Default is GCN.")
    parser.add_argument("--lr",type=float, help="learning rate")
    parser.add_argument("--epochs",type=int, help="number of epochs")
    parser.add_argument("--weights",type=str, help="weights for the weighted loss")
    parser.add_argument("--optimizer",type=str, default="Adam", help="the type of loss")

    parser.add_argument("--loss",type=str, help="the type of loss")

    parser.add_argument("--criterion",type=str, help="the type of loss")
    parser.add_argument("--log_dir",type=str, help="logs directory")



    return parser.parse_args()



def create_model(model_type, input_dim, hidden_dim1, hidden_dim, output_dim, num_layers, hidden_dim_enc=None):
    
    if model_type == 'GAT':
      return GAT(input_dim, hidden_dim1, hidden_dim, output_dim, num_layers)


def choose_optimizer(model, optimizer_type, lr):
  if optimizer_type =="Adam":
    return torch.optim.Adam(model.parameters(), lr=lr)



def criterion(criterion):
  if criterion=="L1loss":
    return torch.nn.L1Loss()
  elif criterion=="MSE":
    return torch.nn.MSELoss() 
  
  elif criterion=="huber":
    return torch.nn.HuberLoss(delta=0.63)
  






def main(args):


  """# Define our Material Limits and Number of Sections (N)"""

  
  Nmaterial=3
  MaterialType=['AC','B','SG'] #AC, base, subbase, subgrade
  Sublayermax=[2,2,1] #each material can have up to 5 sublayers, excluding subgrade


  Thicknessrange=[[2,16],[4,20]] #thickness range in inches
  Modulusrange=[[500,2000],[50,300],[5,50]] #modulus range in ksi
  N=1000 #Number of points
  zpoints=14 #how many points to generate along z
  xpoints=10 #how many points to generate along x
  seed=42
  ############################


  Thicknessincrement=[1,2,4]
  ModulusIncrement=[50,20,20,5]# increment in modulus sampling
  # nurange=[[0,0.499],[0.15,0.499],[0.15,0.499],[0.15,0.499],[0.4,0.499]] #poissons ratio
  # nurange=[[0.3,0.4],[0.2,0.499],[0.2,0.499],[0.2,0.499],[0.2,0.499]] #poissons ratio
  nurange=[[0.3,0.4],[0.2,0.499],[0.2,0.499]] 
  arange=[3,9] #contact radius (in)
  arange=[4,4] #contact radius (in)

  apoints=2 #how many contact radii to analyze
  apoints=1 #how many contact radii to analyze
  factor=0.4
  filter = 2
  split_idx=800
  test_idx=900



  """# Now generate N sections


  """
  FrameLarge_path = args.frame_large_path
  Section_path = args.section_path
   ##analysis paths
  if args.run_analysis:
    nameframe=FrameLarge_path
    namesect=Section_path
    Section,Frame=generatesection(N,Nmaterial,MaterialType,Sublayermax,Thicknessrange,Modulusrange,
                              zpoints,xpoints,Thicknessincrement,ModulusIncrement,nurange,arange,apoints,seed=42)
    FrameLarge, ZS, xs, E,NU,final_dict_ztoE,H,final_dict_ztoH,final_dict_ztonu= generate_query_points(Section, N,xpoints,zpoints,factor,arange,Frame)
    analysis(nameframe, namesect,FrameLarge,Section)
    with open(nameframe, 'rb') as fp:
      FrameLarge = pickle.load(fp)
    with open(namesect, 'rb') as fp:
      Section = pickle.load(fp)


  else:
    with open(args.frame_large_path, 'rb') as fp:
      FrameLarge = pickle.load(fp)
    with open(args.section_path, 'rb') as fp:
      Section = pickle.load(fp)

   
  
  Section_temp,Frame=generatesection(N,Nmaterial,MaterialType,Sublayermax,Thicknessrange,Modulusrange,
                              zpoints,xpoints,Thicknessincrement,ModulusIncrement,nurange,arange,apoints,seed=42)

 


  plot_sample_query_points(Section_temp, xpoints,zpoints,factor)

  FrameLarge_temp, ZS, xs, E,NU,final_dict_ztoE,H,final_dict_ztoH,final_dict_ztonu= generate_query_points(Section, N,xpoints,zpoints,factor,arange,Frame)




  if args.nn:

    TRAIN, TRAIN_out, VAL, VAL_out, TEST, TEST_out, ZS_train, ZS_val, ZS_test,mins_train,maxs_train=train_val_test_generate(FrameLarge,final_dict_ztoE, ZS,xs,split_idx, test_idx,N, final_dict_ztoH,final_dict_ztonu)
    training_nn(args,TRAIN, TRAIN_out, VAL, VAL_out, TEST, TEST_out,Section,ZS,xs,mins_train,maxs_train)

  
    
  elif args.predict == "predict_strain":
    train_df = FrameLarge.loc[FrameLarge['Structure']<=split_idx]
    train_df=train_df[['Strain_Z','Strain_R','Strain_T']]

    train_output = train_df.describe().to_string()
    with open('df_stats_txt', 'w') as file:
      file.write("TRAIN_STATS")
      file.write(train_output)
  
    val_df = FrameLarge.loc[(FrameLarge['Structure'] > split_idx) & (FrameLarge['Structure'] <= test_idx)]  
   

    val_df=val_df[['Strain_Z','Strain_R','Strain_T']]
    val_output = val_df.describe().to_string()
    with open('df_stats_txt','a') as file:
      file.write("VAL_STATS")

      file.write(val_output)
    
    test_df = FrameLarge.loc[FrameLarge['Structure']>test_idx]
    # print(test_df)
    test_df=test_df[['Strain_Z','Strain_R','Strain_T']]


    test_output = test_df.describe().to_string()
  

    ZS, DF =  remove_strain_z(FrameLarge)
    print("SACING")
    DF.to_csv('processed_data.csv', index=False)
    print("df ")

    TRAIN, TRAIN_out, VAL, VAL_out, TEST, TEST_out, ZS_train, ZS_val, ZS_test,mins_train,maxs_train=train_val_test_generate(DF,final_dict_ztoE, ZS,xs,split_idx, test_idx,N, final_dict_ztoH,final_dict_ztonu)
    MAT_edge, MAT_dist, MAT_edge_val, MAT_dist_val, MAT_edge_test, MAT_dist_test=formation_of_matrices(TRAIN,VAL, TEST, ZS_train,xs, ZS_val, ZS_test)
    with open('data_1500_5neigh.pkl','wb') as f:
      pickle.dump({'ZS':ZS,'xs':xs},f)
    print("train set")
   
  input_dim =5
  hidden_dim1=128
  hidden_dim=90
  output_dim=3
  num_layers=10
      
  model = create_model(args.model_type, input_dim, hidden_dim1, hidden_dim, output_dim, num_layers, hidden_dim_enc=None)

  batched_graph = train_graph(TRAIN, TRAIN_out, MAT_edge, MAT_dist)
  # torch.save(batched_graph,'/work/pi_eokte_umass_edu/Urja/batched_graph_5_neigh_zdrop1500.pth')
  batched_graph_val = val_graph(VAL, VAL_out, MAT_edge_val, MAT_dist_val)
  # torch.save(batched_graph_val,'/work/pi_eokte_umass_edu/Urja/batched_graph_val_5_neigh_zdrop1500_check.pth')

  batched_graph_test =  test_graph(TEST, TEST_out, MAT_edge_test, MAT_dist_test)


  optimizer = choose_optimizer(model, args.optimizer, args.lr)

  loss = loss_func(args.loss)

    # Define criterion
  crit = criterion(args.criterion)

  

 
  if args.mode == "train":
 

    training_loop(args, model, optimizer, batched_graph,batched_graph_val, batched_graph_test, args.lr, args.epochs,crit, Section,ZS,xs,weighted_loss=None)

  if args.mode == "eval":
    test_struct = 900
    test_g_struct =0
    
    if args.predict == "predict_strain":
      # with open('/work/pi_eokte_umass_edu/Urja/batched_graph_5_neigh_zdrop1500.pth', 'rb') as fp:
      #   batched_graph = torch.load(fp)
      # with open('/work/pi_eokte_umass_edu/Urja/batched_graph_test_5_neigh_zdrop1500.pth', 'rb') as fp:
      #   batched_graph_test = torch.load(fp)
      evaluation_strain(args,model,batched_graph,mins_train,maxs_train,Section,xs,ZS,batched_graph_test,test_struct,test_g_struct)
if __name__ == "__main__":
  args = parse_arguments()
  

  main(args)

