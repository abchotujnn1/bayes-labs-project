import torch
import math
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Draw import IPythonConsole
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data

def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x==s, allowable_set))

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


class cyp1a2_dataset(Dataset):
    def __init__(self, path):
        with open(path) as file:
            self.data=pd.read_csv(file)
            self.data['num_atom']=[Chem.MolFromSmiles(smile).GetNumAtoms() for smile in self.data['smile_1851']]
            self.data=self.data[self.data['num_atom']<=40]
            self.data=self.data[['smile_1851', 'p450-cyp1a2-Potency']]
            self.data=graph_d(self.data)

    def __len__(self):
        return(len(self.data))

    def __getitem__(self, idx):
        return self.data[idx]

# if __name__=="__main__":
#     path="E:/ind content/bayes_labs_project/deepchem_data/Data_cyp450/data_cyp450_1a2.csv"
#     dataset=cyp1a2_dataset(path)
#     print(len(dataset))
#     print(dataset[0])
