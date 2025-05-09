
import pandas as pd
import numpy as np

from sklearn.preprocessing import PowerTransformer


def frame_filtering(filter,FrameLarge):
    Frame_filtered = pd.DataFrame()

    

    for structure in FrameLarge['Structure'].unique():
        frame = FrameLarge[FrameLarge['Structure']==structure]
        if (frame.Stress_Z[frame.r<0.51]-filter<0).any():
            ind=np.where(frame.Stress_Z[frame.r<0.51]-filter<0)[0][0]
            frame_ind=(frame.Stress_Z[frame.r<0.51]-filter<0).index[ind]
            z_crit=frame.loc[frame_ind,'z']
            frame_filtered=frame.drop(frame[frame['z']>z_crit+0.01].index)
            Frame_filtered = pd.concat([Frame_filtered, frame_filtered], ignore_index=True)

        else:
            Frame_filtered = pd.concat([Frame_filtered, frame], ignore_index=True)
        
    return Frame_filtered

def filtering_ZS(ZS,xs,Frame_filtered):
    ZS_new = []
    ZS_old = ZS.copy()


    Length= Frame_filtered["Structure"].unique()

    # creating input matrix for training
    for structure in Length:

        struct = int(structure)-1
        filtered=Frame_filtered[Frame_filtered["Structure"]==structure]

        # inp=filtered.loc[:,["z","r"]].copy()
        z_val = filtered['z'].unique()
        ZS_new.append(z_val)

    # normalizing
    ZS_new=[ele/20 for ele in ZS_new]
    
    xs=[ele/10 for ele in xs]

    
    
    return ZS_new,xs
"""total inputs and targets

"""
def split_array(array, lengths):
  
  return np.split(array, np.cumsum(lengths)[:-1])

def remove_strain_z(DF):

    ZS_new=[]
    Length= DF["Structure"].unique()

    for structure in Length:
        struct = int(structure) - 1
        filtered = DF[DF["Structure"] == structure]

        inp = filtered.loc[:, ["Strain_Z"]] * 1e6

        # Check if any Strain_Z value is greater than 2000
        if (inp['Strain_Z'] >=1500).any():
            DF = DF[DF["Structure"] != structure]
            continue 
        z_val = filtered['z'].unique()
        ZS_new.append(z_val)

    # normalizing
    ZS_new=[ele/20 for ele in ZS_new]
    # xs=[ele/10 for ele in xs]
     
    
    return ZS_new,DF
        
def train_val_test_generate( DF,final_dict_ztoE, ZS_new, xs,split_idx, test_idx,N,final_dict_ztoH,final_dict_ztonu):
    print("df")
    print(DF["Structure"].nunique())
    Length= DF["Structure"].unique()


    train_length= Length[:split_idx]
    val_length = Length[split_idx:test_idx]
    test_length = Length[test_idx:N]
    ZS_train=ZS_new[:split_idx]
    ZS_val = ZS_new[split_idx:test_idx]
    ZS_test = ZS_new[test_idx:N]
    TRAIN=[]
    TRAIN_out=[]
    VAL=[]
    VAL_out=[]
    TEST=[]
    TEST_out=[]
    
    # creating input matrix for training
    for structure in train_length:
        struct = int(structure)-1
        filtered=DF[DF["Structure"]==structure]

        inp=filtered.loc[:,["z","r"]].copy()


        for idx,col in inp.iterrows():
            z_val = col['z']

            if z_val in final_dict_ztoE[struct]:
               
                inp.loc[idx,"E"]=final_dict_ztoE[struct][z_val]
            if z_val in final_dict_ztoH[struct]:
                inp.loc[idx,"H"]=final_dict_ztoH[struct][z_val]
            if z_val in final_dict_ztonu[struct]:
                inp.loc[idx,"nu"]=final_dict_ztonu[struct][z_val]
            



        inp['z'] = inp['z'] / 20
        inp['r'] = inp['r'] / 10
        total_inputs = inp.values
        TRAIN.append(total_inputs)
        

        out = filtered[['Strain_Z','Strain_R','Strain_T']]*1e6
        
        
        total_targets=out.values

        TRAIN_out.append(total_targets)
    
    
    TRAIN_out_com = np.concatenate(TRAIN_out)
    maxs_train = TRAIN_out_com.max(axis=0)
    mins_train = TRAIN_out_com.min(axis=0)

    TRAIN_out_scaled = (TRAIN_out_com-mins_train)/(maxs_train-mins_train)



    # Original lengths of the arrays in TEST_out
    lengths = [arr.shape[0] for arr in TRAIN_out]

    # Split the scaled array back into the original list structure
    TRAIN_out_scaled = split_array(TRAIN_out_scaled, lengths)
   
   
#creating input matrix for validation
   
   
    for structure in val_length:
        struct = int(structure)-1
        filtered=DF[DF["Structure"]==structure]

        inp=filtered.loc[:,["z","r"]].copy()


        for idx,col in inp.iterrows():

            z_val = col['z']

            if z_val in final_dict_ztoE[struct]:
                inp.loc[idx,"E"]=final_dict_ztoE[struct][z_val]
            
            if z_val in final_dict_ztoH[struct]:
                inp.loc[idx,"H"]=final_dict_ztoH[struct][z_val]
            
            if z_val in final_dict_ztonu[struct]:
                inp.loc[idx,"nu"]=final_dict_ztonu[struct][z_val]
            
            

        inp['z'] = inp['z'] / 20
        inp['r'] = inp['r'] / 10
        total_inputs = inp.values
        VAL.append(total_inputs)
        # print(inputs)

        out = filtered[['Strain_Z','Strain_R','Strain_T']]*1e6
        
        total_targets=out.values
        VAL_out.append(total_targets)
    VAL_out_com = np.concatenate(VAL_out)
    maxs_val = VAL_out_com.max(axis=0)
    mins_val = VAL_out_com.min(axis=0)

    VAL_out_scaled = (VAL_out_com-mins_train)/(maxs_train-mins_train)
        # Original lengths of the arrays in TEST_out
    lengths = [arr.shape[0] for arr in VAL_out]

    # Split the scaled array back into the original list structure
    VAL_out_scaled = split_array(VAL_out_scaled, lengths)
#creating input matrix for validation

    for structure in test_length:
        struct = int(structure)-1
        filtered=DF[DF["Structure"]==structure]

        inp=filtered.loc[:,["z","r"]].copy()

        for idx,col in inp.iterrows():

            z_val = col['z']

            if z_val in final_dict_ztoE[struct]:
                inp.loc[idx,"E"]=final_dict_ztoE[struct][z_val]
            
            if z_val in final_dict_ztoH[struct]:
                inp.loc[idx,"H"]=final_dict_ztoH[struct][z_val]
            
            if z_val in final_dict_ztonu[struct]:
                inp.loc[idx,"nu"]=final_dict_ztonu[struct][z_val]
            
            


        total_inputs = inp.values
        TEST.append(total_inputs)
        # print(inputs)

        out = filtered[['Strain_Z','Strain_R','Strain_T']]*1e6

       
        total_targets=out.values
        TEST_out.append(total_targets)

    inp['z'] = inp['z'] / 20
    inp['r'] = inp['r'] / 10
    TEST_out_com = np.concatenate(TEST_out)
    maxs_test = TEST_out_com.max(axis=0)
    mins_test = TEST_out_com.min(axis=0)

    TEST_out_scaled = (TEST_out_com-mins_train)/(maxs_train-mins_train)

        # Original lengths of the arrays in TEST_out
    lengths = [arr.shape[0] for arr in TEST_out]

    # Split the scaled array back into the original list structure
    TEST_out_scaled = split_array(TEST_out_scaled, lengths)

    return TRAIN, TRAIN_out, VAL, VAL_out, TEST, TEST_out, ZS_train, ZS_val, ZS_test,mins_train,maxs_train

def formation_of_matrices(TRAIN,VAL, TEST, ZS_train,xs, ZS_val, ZS_test):
    #COnsidering multiple inputs
    MAT_edge = []
    MAT_dist=[]
    for i in range(len(TRAIN)):
        inputs_dataset = TRAIN[i]
        inputs_matrix_example = np.reshape(inputs_dataset[:,:2],(len(ZS_train[i]),len(xs),2))

        r= len(inputs_matrix_example)
        c= len(inputs_matrix_example[0])
        adjacent_nodes =[(1,0),(2,0),(3,0),(4,0),(5,0),(0,1),(0,-1),(-1,0),(-2,0),(-3,0),(-4,0),(-5,0)]

        # adjacent_nodes =[(1,0),(0,1),(0,-1),(-1,0)]
        edge_index=[]
        edge_distance=[]

        for i in range(r):
            for j in range(c):
                curr_node = inputs_matrix_example[i][j]

                for dx,dy in adjacent_nodes:
                    m,n=dx+i,dy+j

                    if m>=0 and m<r and n>=0 and n<c:
                        adjacent_node = inputs_matrix_example[m][n]
                        dist = np.linalg.norm(curr_node - adjacent_node)
                        edge_distance.append(1/dist) #flipping the distance since it is used as a weight
                        edge_index.append((i * c + j, m * c + n))

        MAT_edge.append(edge_index)
        MAT_dist.append(edge_distance)
    
    MAT_edge_val = []  #Inputs for validating the model

    MAT_dist_val=[]
    for i in range(len(VAL)):
        inputs_dataset = VAL[i]

        inputs_matrix_example = np.reshape(inputs_dataset[:,:2],(len(ZS_val[i]),len(xs),2))

        r= len(inputs_matrix_example)
        c= len(inputs_matrix_example[0])
        adjacent_nodes =[(1,0),(2,0),(3,0),(4,0),(5,0),(0,1),(0,-1),(-1,0),(-2,0),(-3,0),(-4,0),(-5,0)]

        # adjacent_nodes =[(1,0),(0,1),(0,-1),(-1,0)]
        edge_index=[]
        edge_distance=[]

        for i in range(r):
            for j in range(c):
                curr_node = inputs_matrix_example[i][j]

                for dx,dy in adjacent_nodes:
                    m,n=dx+i,dy+j

                    if m>=0 and m<r and n>=0 and n<c:
                        adjacent_node = inputs_matrix_example[m][n]
                        dist = np.linalg.norm(curr_node - adjacent_node)
                        edge_distance.append(1/dist) #flipping the distance since it is used as a weight
                        edge_index.append((i * c + j, m * c + n))

        MAT_edge_val.append(edge_index)
        MAT_dist_val.append(edge_distance)

    MAT_edge_test = [] #inputs for testing the model

    MAT_dist_test=[]
    for i in range(len(TEST)):
        inputs_dataset = TEST[i]

        inputs_matrix_example = np.reshape(inputs_dataset[:,:2],(len(ZS_test[i]),len(xs),2))

        r= len(inputs_matrix_example)
        c= len(inputs_matrix_example[0])
        # adjacent_nodes =[(1,0),(0,1),(0,-1),(-1,0)]
        adjacent_nodes =[(1,0),(2,0),(3,0),(4,0),(5,0),(0,1),(0,-1),(-1,0),(-2,0),(-3,0),(-4,0),(-5,0)]

        edge_index=[]
        edge_distance=[]

        for i in range(r):
            for j in range(c):
                curr_node = inputs_matrix_example[i][j]

                for dx,dy in adjacent_nodes:
                    m,n=dx+i,dy+j

                    if m>=0 and m<r and n>=0 and n<c:
                        adjacent_node = inputs_matrix_example[m][n]
                        dist = np.linalg.norm(curr_node - adjacent_node)
                        edge_distance.append(1/dist) #flipping the distance since it is used as a weight
                        edge_index.append((i * c + j, m * c + n))

        MAT_edge_test.append(edge_index)
        MAT_dist_test.append(edge_distance)
    return MAT_edge, MAT_dist, MAT_edge_val, MAT_dist_val, MAT_edge_test, MAT_dist_test