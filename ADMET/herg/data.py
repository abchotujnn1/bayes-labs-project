######################################################################################
import torch
import math
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
# % matplotlib inline
import seaborn as sns
import torch.nn.functional as F
from torch import nn, optim
import rdkit
from rdkit import Chem
from rdkit.Chem import DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem import RDConfig
from rdkit.Chem import rdBase
from rdkit.Chem.Draw import IPythonConsole
from torch.utils.data import Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from torch_geometric.data import Data

def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x==s, allowable_set))

def atom_feature(atom):
    symbol_set = ['C', 'N', 'O', 'S', 'F', 'H', 'Si', 'P', 'Cl', 'Br',
                  'Li', 'Na', 'K', 'Mg', 'Ca', 'Fe', 'As', 'Al', 'I', 'B',
                  'V', 'Tl', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn',
                  'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'Mn', 'Cr', 'Pt', 'Hg', 'Pb'] #40
    degree_set = [0,1,2,3,4,5]                          #6
    num_hydrogens_set = [0,1,2,3,4]                     #5
    valency_set = [0,1,2,3,4,5]                         #6+1 last
    return np.array(one_of_k_encoding(atom.GetSymbol(), symbol_set) +
                    one_of_k_encoding(atom.GetDegree(), degree_set) +
                    one_of_k_encoding(atom.GetTotalNumHs(), num_hydrogens_set) +
                    one_of_k_encoding(atom.GetImplicitValence(), valency_set) +
                    [atom.GetIsAromatic()]).astype('float')

def graph_representation(mol,label, max_atoms):
    adj = np.zeros((max_atoms, max_atoms))
    atom_features = np.zeros((max_atoms, 58))
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
    atom_features[0:num_atoms, 0:58] = np.array(features_tmp)
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

def change(x):
    if x=='yes':
        return 1
    return 0
path="E:/ind content/bayes_labs_project/deepchem_data/HERG_DATASET\herg_data.csv"
class NumbersDataset(Dataset):
    def __init__(self, path):
        with open(path) as data:
            self.data = pd.read_csv(data)
            self.data = self.data[['SMILES', 'hERG K+ Channel Blocking {measured}']]
            self.data['hERG K+ Channel Blocking {measured}'] = self.data['hERG K+ Channel Blocking {measured}'].map(change)
            self.data['num_a'] = [Chem.MolFromSmiles(m).GetNumAtoms() for m in self.data['SMILES']]
            self.data = self.data[self.data['num_a'] <= 40]
            self.data=self.data.drop('num_a',axis=1)
            self.data = graph_d(self.data)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]
dataset=NumbersDataset(path)
print(len(dataset)) ####3414
