
#Import packages necessary for analysis
import sys
import os
sys.path.append('..')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from copy import deepcopy
from LayeredElastic.Main.MLEV_Parallel import PyMastic
import pickle



#Function for converting everything to a dataset
def fewpoints(x,y,z,RSO,download):

  columns=['x','y','z']+list(RSO.keys())

  length=np.size(RSO[list(RSO.keys())[0]])
  DF=pd.DataFrame(data=np.zeros([length,len(columns)]),columns=columns)
  counter=0
  for xx in range(len(x)):
    for yy in range(len(y)):
      for zz in range(len(z)):
        DF.loc[counter,'x']=x[xx]
        DF.loc[counter,'y']=y[yy]
        DF.loc[counter,'z']=z[zz]
        for col in columns[3:]:
          mult=1
          if col[:3]=='eps':
            mult=10**6
          DF.loc[counter,col]=RSO[col][yy,xx,zz]*mult
        counter=counter+1
  if download:
    DF.to_excel("PLEA.xlsx")
    files.download('PLEA.xlsx')
  return DF

# for i in range(10):
#     clear_output(wait=True)
#     print("All good, move to the next step")

"""'''
Created on Fri Mar 10 09:36:31 2023


To generate points we have to follow some rules

    1) Select number of materials - between 2 and 4
        - AC only to AC, base, and subbase. Subgrade is always present
    2) For the selected materials select thicknesses
        - With increments of 1 in for AC, 2 in for base, 4 in for subbase
    3) For each material and thickness, decide on number of sublayers
        - Sublayers must be at least 1 in in thickness
        - # of sublayers can be between 1 and sublayermax
        - For now, let's assume equal thickness for all sublayers in a layer
            - If thickness is 6 in and there are 2 sublayers, they are 3 in thick
        - Subgrade have no sublayers
    4) Assign modulus to layers
        - Increments of 50 ksi for AC, 20 for base and subbase, 5 for subgrade
        - sublayer moduli must be decreasing in order
            - if AC has three sub layers and top one has modulus of 500ksi, second to top must be less and so on
            - same for base and subbase assuming stress hardening.
    5) Select load magnitude and contact are randomly

'''

# Define our functions
"""

def generatesection(N,Nmaterial,MaterialType,Sublayermax,Thicknessrange,Modulusrange,
                              zpoints,xpoints,Thicknessincrement,ModulusIncrement,nurange,arange,apoints,seed=42): #generates section based on rules described above
    np.random.seed(seed)
    Section={}
    DataFrame={}
    columns=['Structure','Pressure','ContactRadius','z','r']
    thickCol=[f'H{i}' for i in range(1, sum(Sublayermax))]
    columns.extend(thickCol)
    modCol=[f'E{i}' for i in range(1, sum(Sublayermax)+1)]
    columns.extend(modCol)
    nuCol=[f'nu{i}' for i in range(1, sum(Sublayermax)+1)]
    columns.extend(nuCol)

    for sect in range(N):
        Section[sect]={}
        DataFrame[sect]=np.zeros(len(columns))
        #To generate we have to follow some rules as described above
        #First, decide on number of materials
        MatNum=np.random.randint(2,Nmaterial+1)

        #Then on the thickness
        Thick=[]
        Poisson=[]
        Material=[]
        #for each material, we will have poissons ratio and thickness


        for i in range(MatNum-1):
            T=np.arange(Thicknessrange[i][0],Thicknessrange[i][1]+Thicknessincrement[i],Thicknessincrement[i]) #Increments of 0.5in
            P=np.arange(nurange[i][0],nurange[i][1]+0.001,0.05) #Increments of 0.05
            Thick.append(np.random.choice(T))
            Poisson.append(np.random.choice(P))
            Material.append(MaterialType[i])
        #For subgrade
        P=np.arange(nurange[-1][0],nurange[-1][1]+0.001,0.05)
        Poisson.append(np.random.choice(P))
        Material.append(MaterialType[-1])

        #Round the thickness and poissons ratio
        Thick=np.round(Thick,2)
        Poisson=np.round(Poisson,3)

        #Now we can decide on sublayers
        ThickSub=[]
        MaterialSub=[]
        ModulusSub=[]
        PoissonSub=[]
        for i in range(MatNum-1):
            subs=np.random.randint(1,Sublayermax[i]+1)
            M=np.arange(Modulusrange[i][0],Modulusrange[i][1]+ModulusIncrement[i],ModulusIncrement[i]) #increments of 50ksi
            Modulus0=np.random.choice(M)

            if Thick[i]/subs<1: #if smaller than 1 in, no sublayers
                MaterialSub.append(MaterialType[i])
                ThickSub.append(Thick[i])
                ModulusSub.append(Modulus0)
                PoissonSub.append(Poisson[i])

                continue

            for j in range(subs): #else, divide into sublayers
                MaterialSub.append(MaterialType[i])
                ThickSub.append(Thick[i]/subs)
                PoissonSub.append(Poisson[i])

                if j==0: #if we are at the first sublayer assign modulus 0
                    ModulusSub.append(Modulus0)
                else: #else, assign a smaller modulus
                    Modulus0=np.random.uniform(low=Modulusrange[i][0], high=Modulus0)
                    ModulusSub.append(Modulus0)

        #For subgrade
        ModulusSub.append(np.round(np.random.choice(np.arange(Modulusrange[-1][0],Modulusrange[-1][1]+ModulusIncrement[-1],ModulusIncrement[-1]))))
        PoissonSub.append(Poisson[-1])
        MaterialSub.append('SG')
        #Round the values
        ThickSub=np.round(ThickSub,2)
        ModulusSub=np.round(ModulusSub)
        PoissonSub=np.round(PoissonSub,3)

        #To create the dictionary
        Section[sect]['Material']=Material
        Section[sect]['Thickness']=Thick
        Section[sect]['Poisson']=Poisson
        Section[sect]['MaterialSub']=MaterialSub
        Section[sect]['ThicknessSub']=ThickSub
        Section[sect]['PoissonSub']=PoissonSub
        Section[sect]['ModulusSub']=ModulusSub

        #To create the dataframe
        t=np.append(Section[sect]['ThicknessSub'], np.zeros(sum(Sublayermax)-1-len(Section[sect]['ThicknessSub'])))
        m=np.insert(Section[sect]['ModulusSub'], -1, np.zeros(sum(Sublayermax)-len(Section[sect]['ModulusSub'])))
        p=np.insert(Section[sect]['PoissonSub'], -1, np.zeros(sum(Sublayermax)-len(Section[sect]['PoissonSub'])))
        DataFrame[sect]=np.append(np.zeros(5),t)
        DataFrame[sect][0]=sect+1 #assign structure
        DataFrame[sect][1]=80 #assign pressure of 80 psi (9000/np.pi/6**2)
        DataFrame[sect]=np.append(DataFrame[sect],m)
        DataFrame[sect]=np.append(DataFrame[sect],p)
    Frame=pd.DataFrame.from_dict(DataFrame, orient='index',columns=columns)
    return Section,Frame



"""# Now generate N sections


"""



"""# Generate the query points



"""
def plot_sample_query_points(Section, xpoints, zpoints,factor):
  ind=0
  RS={}
  plt.close('all')

  th=sum(Section[ind]['Thickness'])+12

  zs=np.power(np.linspace(0,np.power(th,factor),zpoints),1/factor) #sampling near the surface


  zs=np.sort(np.append(zs,np.append(np.cumsum(Section[ind]['ThicknessSub'])+0.01,np.cumsum(Section[ind]['ThicknessSub'])-0.01)))


  xs=np.linspace(0,np.sqrt(10),xpoints)**2


  xv, zv = np.meshgrid(xs, zs, indexing='ij')
  # plt.plot(xv, zv, marker='o', color='k', linestyle='none')
  # plt.show()

  factor=0.4
  zs=np.power(np.linspace(0,np.power(th,factor),zpoints),1/factor)
  # plt.plot(zs)


def generate_query_points(Section, N,xpoints,zpoints,factor,arange,Frame):
  
  FrameLarge_temp=[]
  final_dict_ztoE=[]
  final_dict_ztoH=[]
  final_dict_ztonu = []
  ZS=[]
  
  H=[]
  E=[]
  NU = []


  
  for i in range(N):
    dict_z_to_E = {}
    dict_z_to_H = {}
    dict_z_to_nu={}
    th=sum(Section[i]['Thickness'])+12
    zs=np.power(np.linspace(np.sqrt(0.5),np.power(th,factor),zpoints),1/factor) #sampling near the surface

    zs=np.sort(np.append(zs,np.append(np.cumsum(Section[i]['ThicknessSub'])+0.01,np.cumsum(Section[i]['ThicknessSub'])-0.01)))
    ZS.append(zs)

    E_per_section=np.zeros(len(zs))
    H_per_section = np.zeros(len(zs))
    nu_per_section = np.zeros(len(zs))
    points_above_boundary=np.cumsum(Section[i]['ThicknessSub'])+0.01
    points_below_boundary=np.cumsum(Section[i]['ThicknessSub'])-0.01

    j=0
    while j<=len(Section[i]['ModulusSub'])-1:
      if j==0:

        E_per_section[(zs<=points_below_boundary[j])]=Section[i]['ModulusSub'][j]
        H_per_section[(zs<=points_below_boundary[j])]=Section[i]['ThicknessSub'][j]
        nu_per_section[(zs<=points_below_boundary[j])]=Section[i]['PoissonSub'][j]
        ind=0
        j+=1
      elif j>0 and j<len(Section[i]['ModulusSub'])-1:

        E_per_section[(zs>=points_above_boundary[ind])&(zs<=points_below_boundary[j])]=Section[i]['ModulusSub'][j]
        H_per_section[(zs>=points_above_boundary[ind])&(zs<=points_below_boundary[j])]=Section[i]['ThicknessSub'][j]
        nu_per_section[(zs>=points_above_boundary[ind])&(zs<=points_below_boundary[j])]=Section[i]['PoissonSub'][j]
         
        ind+=1

        j+=1
      elif j>=len(Section[i]['ModulusSub'])-1:

        E_per_section[(zs>=points_above_boundary[ind])]=Section[i]['ModulusSub'][j]
        H_per_section[(zs>=points_above_boundary[ind])]=-1
        nu_per_section[(zs>=points_above_boundary[ind])]=Section[i]['PoissonSub'][j]
        j+=1

    for z,h in zip(zs,H_per_section):
      dict_z_to_H[z] = h/100
    final_dict_ztoH.append(dict_z_to_H)
    for z,e in zip(zs,E_per_section):
      dict_z_to_E[z]=e/100
    final_dict_ztoE.append(dict_z_to_E)
    for z,nu in zip(zs,nu_per_section):
      dict_z_to_nu[z] = nu
    final_dict_ztonu.append(dict_z_to_nu)


    H.append(H_per_section)
    E.append(E_per_section)
    NU.append(nu_per_section)
    xs = np.linspace(np.sqrt(0.5), np.sqrt(10), xpoints)**2
    #radi=np.sort(random.sample(range(arange[0],arange[1]),apoints))
    radi=[arange[0]]
    Section[i]['z']=zs
    Section[i]['x']=xs
    Section[i]['a']=radi
    FrameTemp=deepcopy(Frame.loc[i:i,:])
    FrameTemp=pd.DataFrame(np.repeat(FrameTemp.values, len(radi)*len(zs)*len(xs), axis=0))
    FrameTemp.columns = Frame.columns
    res = np.matrix([[ii, j, k] for ii in radi
                  for j in zs
                  for k in xs])
    FrameTemp.iloc[:,2:5]=res
    FrameLarge_temp.append(FrameTemp)
  FrameLarge_temp = pd.concat(FrameLarge_temp)
  FrameLarge_temp=FrameLarge_temp.reset_index(drop=True)
  FrameLarge_temp[['Displacement_Z', 'Displacement_H', 'Stress_Z', 'Stress_R', 'Stress_T', 'Stress_RZ', 'Strain_Z', 'Strain_R', 'Strain_T']]=0
  return FrameLarge_temp, ZS, xs, E,NU,final_dict_ztoE,H,final_dict_ztoH,final_dict_ztonu

def analysis(nameframe, namesect,FrameLarge,Section):

  ##I think this code is the correct code
  first=(FrameLarge.Displacement_Z.values == 0).argmax() #find the first zero of the sections that are not run
  first=FrameLarge.Structure[first]
  print(first)
  for ind in range(int(first)-1,len(Section)): #start from the one with no analysis
    print('Section',ind)
    Section[ind]['Response']={}
    for j in range(1): #loop through contact radius (reduced to two)
      print('Contact area',Section[ind]['a'][j])
      RS=PyMastic(80,Section[ind]['a'][j],Section[ind]['x'],Section[ind]['z'],Section[ind]['ThicknessSub'],Section[ind]['ModulusSub']*1000,Section[ind]['PoissonSub'], ZRO=7*1e-20, isBounded = np.ones(len(Section[ind]['ModulusSub'])), iteration = 1600, inverser = 'solve',tol=0.0001,every=10,verbose=True)
      Section[ind]['Response'][j]=RS
      zero=(FrameLarge.Displacement_Z.values == 0).argmax()
      for k in range(len(RS.keys())): #save the results to the dataframe
        keyy=list(RS.keys())[k]
        resp=RS[keyy].flatten()
        FrameLarge.loc[zero:zero+len(resp)-1,keyy]=resp
    if ind%50==0 and ind>0:
      with open(namesect, "wb") as fp:
        pickle.dump(Section, fp)  # encode dict into JSON
      with open(nameframe, "wb") as fp:
        pickle.dump(FrameLarge, fp)  # encode dict into JSON
      print('Saved')
  with open(namesect, "wb") as fp:
    pickle.dump(Section, fp)  # encode dict into JSON
  with open(nameframe, "wb") as fp:
    pickle.dump(FrameLarge, fp)  # encode dict into JSON