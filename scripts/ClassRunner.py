import geopandas as gpd
import numpy as np
import torch
import json
from tqdm import tqdm

import sys, os
from .ClassNPYdatasetGEN2 import NPYdataset

from torch.utils.data import DataLoader


class Runner:
    def __init__(self,
                 path_to_test_table,
                 path_to_dataset):
        self.path_to_test_table = path_to_test_table
        self.path_to_dataset = path_to_dataset
    
    def run_inference(self,
                      model,
                      path_to_loader_config,
                      path_to_modis_availible_days = './modis_features_days.json',
                      batch_size=16,
                      n_jobs=-1,
                      device='cuda'):

        model.to(device).eval()
        
        dataset_params = json.load(open(path_to_loader_config, 'r'))
        dataset_params.update({"dataset_path" : self.path_to_dataset,
                               "path_to_samples_table" : self.path_to_test_table,
                               "mode" : 'binary',
                               "weather_data" : "era5",
                               "path_to_modis_availible_days" : path_to_modis_availible_days})
        dataset = NPYdataset(dataset_params)
        
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=n_jobs)
    
        all_outs = []
        all_targets = []
        for batch in tqdm(dataloader):
            if len(batch)==4:
                static, dynamic, additional, targets = batch
                additional = additional.to(device)
                static = static.to(device)
                dynamic = dynamic.to(device)
                out = model(static, dynamic, additional)
                
            elif len(batch)==3:
                static, dynamic, targets = batch
                static = static.to(device)
                dynamic = dynamic.to(device)
                out = model(static, dynamic)
            else:
                raise RuntimeError

            all_outs.append(out.detach().cpu())
            all_targets.append(targets.detach().cpu())
            
        all_outs = torch.cat(all_outs, axis=0).detach().cpu()
        all_targets = torch.cat(all_targets, axis=0).detach().cpu()
        
        return {
            'pred' : all_outs,
            'target' : all_targets,
        }

    @staticmethod
    def run_default_metrics(pred, target, threshold, metrics):
        pred_bool = (pred>threshold)
        out = {}
        for metric in metrics:
            out.update({metric : metrics[metric](pred_bool, target)})

        return out


















        
        