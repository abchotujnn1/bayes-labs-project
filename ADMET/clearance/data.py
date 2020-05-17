###################Importing the requires library###############
import torch
import math
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem
from torch.utils.data import Dataset
from torch_geometric.data import Data
###############################one-hot encoding for every feature of atom##############
def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x==s, allowable_set))
############################generating atomic features of atom#########################
def atom_feature(atom):
    symbol_set = ['C', 'N', 'O', 'S', 'F', 'H', 'P', 'Cl', 'Br', 'K', 'Mg','Si']  # 12
    # degree_set = [0, 1, 2, 3, 4, 5]  # 6
    num_hydrogens_set = [0, 1, 2, 3, 4]  # 5
    valency_set = [0, 1, 2, 3, 4, 5]  # 6
    formal_charge_list = [-3, -2, -1, 0, 1, 2, 3]  # 7
    hybridization_list = [Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
                          Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
                          Chem.rdchem.HybridizationType.SP3D2]  # 5
    number_radical_e_list = [0, 1, 2]  # 3
    # chirality = ['R', 'S'] #1
    # Aromatic              #1
    mol_wt=[12.011, 14.007, 15.99, 32.065,18.998,1.007,30.97,35.453, 79.904, 39.098, 24.305,28.085]
    m=np.mean(mol_wt)
    st_d=np.std(mol_wt)
    std_mol_wt=[(i-m)/st_d for i in mol_wt]


    return np.array(list(np.multiply(one_of_k_encoding(atom.GetSymbol(), symbol_set), std_mol_wt)) + ####12
                                                                                                     # one_of_k_encoding(atom.GetDegree(), degree_set) +
                    one_of_k_encoding(atom.GetTotalNumHs(), num_hydrogens_set) +                   ######5
                    one_of_k_encoding(atom.GetImplicitValence(), valency_set) +                    ######6
                    one_of_k_encoding(atom.GetFormalCharge(), formal_charge_list) +                ######7
                    one_of_k_encoding(atom.GetHybridization(), hybridization_list) +               ######5
                    one_of_k_encoding(atom.GetNumRadicalElectrons(), number_radical_e_list) +      ######3
                    [atom.GetIsAromatic()] + [atom.HasProp('_ChiralityPossible')]).astype('float') ######1+1
################################## Representing one molecule to graph data ###########################
def graph_representation(mol,label, max_atoms):
    adj = np.zeros((max_atoms, max_atoms))
    atom_features = np.zeros((max_atoms, 40))
    num_atoms = mol.GetNumAtoms()
    adj[0:num_atoms, 0:num_atoms] = Chem.rdmolops.GetAdjacencyMatrix(mol)
    edge0 = []
    edge1 = []
    for i, l in enumerate(adj):
        for j, k in enumerate(l):
            if (k == 1):
                edge0.append(i)
                edge1.append(j)
    edge_idx=[edge0,edge1]
    features_tmp = []
    for atom in mol.GetAtoms():
        features_tmp.append(atom_feature(atom))
    atom_features[0:num_atoms, 0:40] = np.array(features_tmp)
    return edge_idx, atom_features,label

############################ total data to graph data #######################
def graph_d(smiles):
    graph_data=[]
    for i,j in smiles.values:
        mol=Chem.MolFromSmiles(i)
        edge_index, x, l=graph_representation(mol, j, 40)
        edge_index=torch.tensor(edge_index,dtype=torch.long)
        x=torch.tensor(x,dtype=torch.float)
        l=torch.tensor(l,dtype=torch.float).view(1,-1)
        d=Data(x=x,edge_index=edge_index,y=l)
        graph_data.append(d)
    return graph_data
######################upper label fixing for the lebel #####################
def u_label(x):
    if(x>=500):
        return 500
    return x



###################  Preparing the dataloader for dataset  ###############################
class clearance(Dataset):
    def __init__(self, clr_path):
        with open(clr_path) as clr_data:
            self.data = pd.read_csv(clr_data, delimiter=";")
            self.data_f = self.data[['Canonical_Smiles', 'hlm_clearance[mL.min-1.g-1]']]
            self.data_f = self.data_f.dropna()
            # target=[math.log(float(re.sub(",", ".", str(i)))) for i in self.data_f['hlm_clearance[mL.min-1.g-1]']]
            target = [float(re.sub(",", ".", str(i))) for i in self.data_f['hlm_clearance[mL.min-1.g-1]']]
            target = list(map(u_label, target))
            print(np.mean(target))
            print(np.std(target))

            target = [math.log(k) for k in target]
            m = np.mean(target)
            st_d = np.std(target)
            target = [(i - m) / st_d for i in target]
            print(np.mean(target))
            print(np.std(target))
            self.data_f = self.data_f.drop('hlm_clearance[mL.min-1.g-1]', axis=1)
            self.data_f['label'] = target
            self.data_f['num_a'] = [Chem.MolFromSmiles(m).GetNumAtoms() for m in self.data_f['Canonical_Smiles']]
            self.data_f = self.data_f[self.data_f['num_a'] <= 40]
            self.data_f = self.data_f.drop('num_a', axis=1)
            self.data_f = graph_d(self.data_f)

    def __len__(self):
        return len(self.data_f)

    def __getitem__(self, idx):
        return self.data_f[idx]




if __name__=="__main__":
    clr_path = "E:/ind content/bayes_labs_project/deepchem_data/Dataset_chembl_clearcaco.txt"
    dataset = clearance(clr_path)
    print(dataset[0])
    print(len(dataset))
#     d_train = dataset[:4070]
#     d_test = dataset[4070:]
# ################# Visualizing the distribution of train and test target value ####################
#     l1 = [float(i.y) for i in d_train]
#     l2 = [float(i.y) for i in d_test]
#     print(np.mean(l1))
#     print(np.std(l1))
#     print(np.mean(l2))
#     print(np.std(l2))
#     # plt.figure(1)
#     # plt.subplot(221)
#     # plt.hist(l1, range=[-5, 5], bins=20)
#     # plt.subplot(222)
#     # sns.distplot(l1)
#     # plt.subplot(223)
#     # plt.hist(l2, range=[-5, 5], bins=20)
#     # plt.subplot(224)
#     # sns.distplot(l2)
#     # plt.show()
