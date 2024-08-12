import pennylane as qml
import numpy as np
import random
import matplotlib.pyplot as plt
import csv
import pandas as pd
import argparse
import os
import math
import datetime
import time
from frechetdist import frdist

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.utils.data as data
import torch.nn as nn
import torch.nn.functional as F
import torch

from solver import Solver
from data_loader import get_loader
from torch.backends import cudnn
from utils import *
from models import Generator, Discriminator
from data.sparse_molecular_dataset import SparseMolecularDataset
from rdkit import Chem


def str2bool(v):
    return v.lower() in ('true')

qubits = 10
# Set up your ibmq credentials first from https://quantum-computing.ibm.com/
demo_on_ibmq = False

if demo_on_ibmq:
    dev = qml.device('qiskit.ibmq', wires=qubits, backend='ibmq_16_melbourne')
else:
    dev = qml.device('default.qubit', wires=qubits)

@qml.qnode(dev, interface='torch')
def gen_circuit(w):
    # random noise as generator input
    z1 = random.uniform(-1, 1)
    z2 = random.uniform(-1, 1)
    layers = 1    
    
    # construct generator circuit for both atom vector and node matrix
    for i in range(qubits):
        qml.RY(np.arcsin(z1), wires=i)
        qml.RZ(np.arcsin(z2), wires=i)
        
    for l in range(layers):
        for i in range(qubits):
            qml.RY(w[i], wires=i)
        for i in range(qubits-1):
            qml.CNOT(wires=[i, i+1])
            qml.RZ(w[i+qubits], wires=i+1)
            qml.CNOT(wires=[i, i+1])
    return [qml.expval(qml.PauliZ(i)) for i in range(qubits)]


parser = argparse.ArgumentParser()

# Quantum circuit configuration
parser.add_argument('--quantum', type=bool, default=True, help='choose to use quantum gan with hybrid generator')
parser.add_argument('--patches', type=int, default=1, help='number of quantum circuit patches')
parser.add_argument('--layer', type=int, default=3, help='number of repeated variational quantum layer')
parser.add_argument('--qubits', type=int, default=10, help='number of qubits and dimension of domain labels')

# Model configuration.
parser.add_argument('--z_dim', type=int, default=8, help='dimension of domain labels')
parser.add_argument('--g_conv_dim', default=[128], help='number of conv filters in the first layer of G')
parser.add_argument('--d_conv_dim', type=int, default=[[128, 64], 128, [128, 64]], help='number of conv filters in the first layer of D')
parser.add_argument('--g_repeat_num', type=int, default=6, help='number of residual blocks in G')
parser.add_argument('--d_repeat_num', type=int, default=6, help='number of strided conv layers in D')
parser.add_argument('--lambda_cls', type=float, default=1, help='weight for domain classification loss')
parser.add_argument('--lambda_rec', type=float, default=10, help='weight for reconstruction loss')
parser.add_argument('--lambda_gp', type=float, default=10, help='weight for gradient penalty')
parser.add_argument('--post_method', type=str, default='softmax', choices=['softmax', 'soft_gumbel', 'hard_gumbel'])

# Training configuration.
parser.add_argument('--batch_size', type=int, default=16, help='mini-batch size')
parser.add_argument('--num_iters', type=int, default=5000, help='number of total iterations for training D')
parser.add_argument('--num_iters_decay', type=int, default=2500, help='number of iterations for decaying lr')
parser.add_argument('--g_lr', type=float, default=0.0001, help='learning rate for G')
parser.add_argument('--d_lr', type=float, default=0.0001, help='learning rate for D')
parser.add_argument('--dropout', type=float, default=0., help='dropout rate')
parser.add_argument('--n_critic', type=int, default=5, help='number of D updates per each G update')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam optimizer')
parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam optimizer')
parser.add_argument('--resume_iters', type=int, default=None, help='resume training from this step')

# Test configuration.
parser.add_argument('--test_iters', type=int, default=5000, help='test model from this step')

# Miscellaneous.
parser.add_argument('--num_workers', type=int, default=1)
parser.add_argument('--mode', type=str, default='test', choices=['train', 'test'])
parser.add_argument('--use_tensorboard', type=str2bool, default=False)

# Directories.
parser.add_argument('--mol_data_dir', type=str, default='data/qm9_5k.sparsedataset')
parser.add_argument('--log_dir', type=str, default='qgan-hg-mr-q10-l3/logs')
parser.add_argument('--model_save_dir', type=str, default='qgan-hg-mr-q10-l3/models')
parser.add_argument('--sample_dir', type=str, default='qgan-hg-mr-q10-l3/samples')
parser.add_argument('--result_dir', type=str, default='qgan-hg-mr-q10-l3/results')

# Step size.
parser.add_argument('--log_step', type=int, default=10)
parser.add_argument('--sample_step', type=int, default=1000)
parser.add_argument('--model_save_step', type=int, default=1000)
parser.add_argument('--lr_update_step', type=int, default=500)



config = parser.parse_known_args()[0]
print(config)

self = Solver(config)



# Inference of generated molecules
start_iters = 0
self.resume_iters = 2715

if self.resume_iters:
    start_iters = self.resume_iters
    self.restore_model(self.resume_iters)

# Get gen_Weights from final line of molgan_red_weights.csv
gen_weights = torch.tensor([1.0403185179516392,-0.9184817921680926,-0.5441340260188031,-1.02346108465704,-3.0797580007724434,-1.4357635241228668,-1.1438467727804762,1.676568628963222,0.03637897371131086,0.353781384630498,-0.1989807691579263,1.978176923837451,1.2839688436569185,2.0091789023064037,1.157357807244109,1.7151845612128107,1.33382449971489,1.6053759461419161,-2.6096406101337766,3.039732822624212,1.2607997525466041,2.929853421744336,0.5944505561489524,-0.5250574391819369,-0.3020537187251504,-0.9655658939580611,1.766146374759935,2.300374914734266,2.969515516099687,0.20828465308817767,0.665425784586327,2.7036490498680683,1.8108200541160677,-0.25419821099919115,-0.6382208693616578,-0.4006212788447594,-2.046187145707842,-1.0392638726714774,-1.2318185376689361,0.21821887659058614,-1.8465778853739854,-2.1018148278250166,0.5545893814873497,-0.1316450966427456,-2.7326518532033353,0.8424558686709491,-0.20267550388638522,1.1355993304295167,-1.3482984208244102,-1.253685094838629,-1.7453157212409112,0.3575429013392557,-2.489718489597295,0.6343534513422875,-1.3245794417868302,2.6927240869576696,0.4588313017482535], requires_grad=False)

# Initialize accumulators for the metrics
# Initialize accumulators for the metrics
total_loss_fake = 0.0
total_loss_value = 0.0
total_fd_bond_only = 0.0
total_fd_bond_atom = 0.0
total_scores = {
    "NP score": 0.0,
    "QED score": 0.0,
    "logP score": 0.0,
    "SA score": 0.0,
    "diversity score": 0.0,
    "drugcandidate score": 0.0,
    "valid score": 0.0,
    "unique score": 0.0,
    "novel score": 0.0
}
count_sa_score = 0  # Counter for SA score to handle NaN values

num_iterations = 1000
for iteration in range(num_iterations):
    sample_list = [gen_circuit(gen_weights) for i in range(self.batch_size)]
    z = torch.stack([torch.stack(batch) for batch in sample_list]).to(self.device).float()


    # Start inference.
    print('Start inference...')
    start_time = time.time()

    mols, _, _, a, x, _, _, _, _ = self.data.next_train_batch(self.batch_size)

    # =================================================================================== #
    #                             1. Preprocess input data                                #
    # =================================================================================== #

    a = torch.from_numpy(a).to(self.device).long()            # Adjacency.
    x = torch.from_numpy(x).to(self.device).long()            # Nodes.
    a_tensor = self.label2onehot(a, self.b_dim)
    x_tensor = self.label2onehot(x, self.m_dim)

    # Z-to-target
    edges_logits, nodes_logits = self.G(z)
    # Postprocess with Gumbel softmax
    (edges_hat, nodes_hat) = self.postprocess((edges_logits, nodes_logits), self.post_method)
    logits_fake, features_fake = self.D(edges_hat, None, nodes_hat)
    g_loss_fake = - torch.mean(logits_fake)

    # Real Reward
    rewardR = torch.from_numpy(self.reward(mols)).to(self.device)
    # Fake Reward
    (edges_hard, nodes_hard) = self.postprocess((edges_logits, nodes_logits), 'hard_gumbel')
    edges_hard, nodes_hard = torch.max(edges_hard, -1)[1], torch.max(nodes_hard, -1)[1]
    mols = [self.data.matrices2mol(n_.data.cpu().numpy(), e_.data.cpu().numpy(), strict=True)
            for e_, n_ in zip(edges_hard, nodes_hard)]
    rewardF = torch.from_numpy(self.reward(mols)).to(self.device)

    # Value loss
    value_logit_real,_ = self.V(a_tensor, None, x_tensor, torch.sigmoid)
    value_logit_fake,_ = self.V(edges_hat, None, nodes_hat, torch.sigmoid)
    g_loss_value = torch.mean((value_logit_real - rewardR) ** 2 + (
                            value_logit_fake - rewardF) ** 2)

    R=[list(a[i].reshape(-1))  for i in range(self.batch_size)]
    F=[list(edges_hard[i].reshape(-1))  for i in range(self.batch_size)]
    fd_bond_only = frdist(R, F)

    R=[list(x[i]) + list(a[i].reshape(-1))  for i in range(self.batch_size)]
    F=[list(nodes_hard[i]) + list(edges_hard[i].reshape(-1))  for i in range(self.batch_size)]
    fd_bond_atom = frdist(R, F)

    loss = {}
    loss['G/loss_fake'] = g_loss_fake.item()
    loss['G/loss_value'] = g_loss_value.item()
    loss['FD/fd_bond_only'] = fd_bond_only
    loss['FD/fd_bond_atom'] = fd_bond_atom

    # Print out training information.
    et = time.time() - start_time
    et = str(datetime.timedelta(seconds=et))[:-7]
    log = "Elapsed [{}], Iteration [{}/{}]".format(et, start_iters, self.num_iters)

    # Log update
    m0, m1 = all_scores(mols, self.data, norm=True)     # 'mols' is output of Fake Reward
    m0 = {k: np.array(v)[np.nonzero(v)].mean() for k, v in m0.items()}
    m0.update(m1)
    loss.update(m0)
    for tag, value in loss.items():
        log += ", {}: {:.4f}".format(tag, value)
    print(log)

    # Accumulate the metrics
    total_loss_fake += loss['G/loss_fake']
    total_loss_value += loss['G/loss_value']
    total_fd_bond_only += loss['FD/fd_bond_only']
    total_fd_bond_atom += loss['FD/fd_bond_atom']

    for key in total_scores.keys():
        if key == "SA score":
            if not np.isnan(loss[key]):
                total_scores[key] += loss[key]
                count_sa_score += 1
        else:
            total_scores[key] += loss[key]

avg_loss_fake = total_loss_fake / num_iterations
avg_loss_value = total_loss_value / num_iterations
avg_fd_bond_only = total_fd_bond_only / num_iterations
avg_fd_bond_atom = total_fd_bond_atom / num_iterations

# Compute average scores, handling `nan` for SA score
avg_scores = {}
for key in total_scores.keys():
    if key == "SA score" and count_sa_score > 0:
        avg_scores[key] = total_scores[key] / count_sa_score
    else:
        avg_scores[key] = total_scores[key] / num_iterations

# Print the averaged metrics
print(f"Averaged over {num_iterations} iterations:")
print(f"Average G/loss_fake: {avg_loss_fake:.4f}")
print(f"Average G/loss_value: {avg_loss_value:.4f}")
print(f"Average FD/fd_bond_only: {avg_fd_bond_only:.4f}")
print(f"Average FD/fd_bond_atom: {avg_fd_bond_atom:.4f}")
for key, value in avg_scores.items():
    print(f"Average {key}: {value:.4f}")


# Keep only valid moleculues evaluated by RDKit
valid_mols = [i for i in mols if i != None]

from rdkit.Chem.Draw import SimilarityMaps
import matplotlib

for mol in valid_mols:
    AllChem.ComputeGasteigerCharges(mol)
    contribs = [mol.GetAtomWithIdx(i).GetDoubleProp('_GasteigerCharge') for i in range(mol.GetNumAtoms())]
    fig = SimilarityMaps.GetSimilarityMapFromWeights(mol, contribs, colorMap=None,  contourLines=10)
    fig.savefig(f"similarity_map2_.png", dpi=300, bbox_inches='tight')