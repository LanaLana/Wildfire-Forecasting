import torch
import os 
import zipfile
import numpy as np


def smart_binary_search(arr, val, atol=1e-6):
    out_idx = np.searchsorted(arr, val)
    if out_idx+1 < len(arr):
        if np.isclose(arr[out_idx+1], val, atol=atol):
            return out_idx+1
    if out_idx-1 >= 0:
        if np.isclose(arr[out_idx-1], val, atol=atol):
            return out_idx-1
    return(out_idx)

def get_total_params(model):
    total_params = sum(
    param.numel() for param in model.parameters()
    )
    return(total_params)


def save_model(model, path_to_save_model, name):
    os.makedirs(path_to_save_model, exist_ok=True)
    torch.save(model.state_dict(), path_to_save_model + "/" + name) 


def load_pretrainned(model, path_to_weights):
    model.load_state_dict(torch.load(path_to_weights))


def unzip(path_in, path_out):
    with zipfile.ZipFile(path_in, 'r') as zip_ref:
        zip_ref.extractall(path_out)


#returns first smaller element
def binary_search(data, val):
    length = len(data)
    r = 0
    l = length-1

    while l>r:
        m = (l+r)//2
        if (m==l) or (m==r):
            break
            
        if data[m]<val:
            r = m
        else:
            l = m
        
    if val<data[m]:
        return(m)
    else: 
        return(m+1)