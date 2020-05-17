import torch
import rdkit
from rdkit import*
import cyp450_1a2_data
from cyp450_1a2_data import*
model=torch.load('cyp450_1a2.pt')
model=model.eval()
s1="CC(=O)[C@H]1CC[C@H]2[C@@H]3CCC4=CC(=O)CC[C@]4(C)[C@H]3CC[C@]12C"#3.87
s2="CO[C@]12[C@H]3N[C@H]3CN1C4=C([C@H]2COC(=O)N)C(=O)C(=C(C)C4=O)N"#-1.6
s3="CCN1CCCC1CNC(=O)c2cc(ccc2OC)S(=O)(=O)N"#0.42
s4="CN1CC(=O)N=C1N"#-3

def drug_to_mol(s1):
    mol=Chem.MolFromSmiles(s1)
    edge_idx,x,l=graph_representation(mol,4,40)
    edge_idx=torch.tensor(edge_idx,dtype=torch.long)
    x=torch.tensor(x,dtype=torch.float)
    l=torch.tensor(l,dtype=torch.float).view(1,-1)
    d=Data(x=x,edge_index=edge_idx,y=l)
    return d
test_drug=drug_to_mol(s1)
output=model(test_drug.to('cuda'))
print(output)
