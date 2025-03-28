import warnings
import os, sys
import re
import json
import numpy as np
import torch
from torch.nn.functional import pad
import rasterio
from rasterio.coords import BoundingBox
import geopandas as gpd
from tqdm import tqdm

import time

from .utils import binary_search


class NPYdataset:
    def __init__(self, params):
        self.params = params
        self._read_params()

    
    def _read_params(self):
        self.dataset_path = self.params['dataset_path']
        self.samples_table = gpd.read_file(self.params['path_to_samples_table'])
        #'binary'/'binary_first_day' or 'multy_days' or 'days' or 'day_classification' or 'days group'
        #'test_features'
        self.LC_mode = self.params.get('LC_mode', 'binary_vector')
        self.mode = self.params.get('mode', None)
        assert self.mode
        self.day_seq_length = self.params.get('day_seq_length', 7)
        assert self.day_seq_length > 1
        self.firecast_seq_length = self.params.get('forecast_seq_length', 0)
        assert self.firecast_seq_length >= 0
        self.weather_data = self.params.get("weather_data", None)
        self.with_permafrost = self.params.get("with_permafrost", False)
        assert self.weather_data
        self.len = len(self.samples_table)
        self.path_to_modis_availible_days = self.params.get('path_to_modis_availible_days', None)
        modis_availible_days_dict = json.load(open(self.path_to_modis_availible_days))
        self.days_8_list = modis_availible_days_dict['_Fpar_500m_']
        self.days_16_list = modis_availible_days_dict['_500m_16_days_NDVI_']
        self.raster_coords = {}
        self.augmenter = self.params.get('augmenter', None)
        self.sample_with_date = self.params.get('sample_with_date', None)
        self.sample_with_coords = self.params.get('sample_with_coords', None)
        self.sample_raster_size = self.params.get('sample_raster_size', (32, 32))
        self.load_only_2022 = self.params.get('load_only_2022', False)
    
    
    def __len__(self):
        return self.len
    
    
    def upload_region_raster_coords(self, reg):
        if reg not in self.raster_coords:
            npz_data = np.load(f'{self.dataset_path}/{reg}/raster_data.npz')
            self.raster_coords.update({
                reg : {"latitude" : npz_data['latitude'],
                       "longitude" : npz_data['longitude']}
            })
        else:
            pass     
    

    @staticmethod
    def features_nan_hook(data, replacement):
        data_tmp = 1*data
        if np.isnan(data).any():
            #nan_args = np.argwhere(np.isnan(data_tmp))
            if replacement is not None:
                data_tmp = np.where(np.isnan(data_tmp), replacement, data_tmp)
            else:
                data_tmp = np.where(np.isnan(data_tmp), -1, data_tmp)
        return data_tmp


    def get_dynamic_features(self, reg, year, day, seq_length, cell_bounds):
        path_to_temporary_daily = f'{self.dataset_path}/{reg}/temporary_daily_era5/{year}'
        
        dynamic = [None,]
        
        ### upload historic weather data
        hist_days = range(day - self.day_seq_length, day)
        for d in hist_days:
            path = f'{path_to_temporary_daily}/d{d:03d}.npy'
            np_file = np.load(path, mmap_mode='r')
            day_data = np_file[:, cell_bounds.top:cell_bounds.bottom+1,
                               cell_bounds.left:cell_bounds.right+1]
            day_data = self.features_nan_hook(day_data, dynamic[-1])
            dynamic.append(day_data)
        
        ### upload forecast weather data
        forecast_days = range(day, day + self.firecast_seq_length)
        for d in forecast_days:
            d = min(304, d)
            path = f'{path_to_temporary_daily}/d{d:03d}.npy'
            np_file = np.load(path, mmap_mode='r')
            day_data = np_file[:, cell_bounds.top:cell_bounds.bottom+1,
                               cell_bounds.left:cell_bounds.right+1]
            day_data = self.features_nan_hook(day_data, dynamic[-1])
            dynamic.append(day_data)

        dynamic = np.stack(dynamic[1:], axis=0)
        return dynamic


    @staticmethod
    def get_raster_coords_extended(latitudes, longitudes, center, cell_size):
        #print("latitudes.min(), latitudes.max(), center[0]", latitudes.min(), latitudes.max(), center[0])
        #print("longitudes.min(), longitudes.max(), center[1]", longitudes.min(), longitudes.max(), center[1])

        r_max = len(longitudes) - 1
        b_max = len(latitudes) - 1
        
        raster_center_t_b = b_max - binary_search(np.flip(latitudes), center[0])
        raster_center_l_r = binary_search(longitudes, center[1])-1
        
        #print(latitudes.shape, raster_center_t_b)
        #print(longitudes.shape, raster_center_l_r)
        
        l = raster_center_l_r - cell_size[0]//2 + 1
        r = raster_center_l_r + cell_size[0] - cell_size[0]//2

        t = raster_center_t_b - cell_size[1]//2 + 1
        b = raster_center_t_b + cell_size[1] - cell_size[1]//2

        cell_bounds = (max(0, l), min(r, r_max), max(0, t), min(b, b_max))
        padding_bounds = (cell_bounds[0]-l, r-cell_bounds[1], cell_bounds[2]-t, b-cell_bounds[3])

        #print(l, r, t, b)
        #print(cell_bounds)
        #print(padding_bounds)

        cell_bounds = BoundingBox(left=cell_bounds[0],
                                  right=cell_bounds[1],
                                  top=cell_bounds[2],
                                  bottom=cell_bounds[3],)
        
        return cell_bounds, padding_bounds
    
    
    @staticmethod
    def get_availible_modis_day(day, availible_days):
        day_to_use = availible_days[0]
        for available_day in availible_days:
            if available_day < day:
                day_to_use = available_day
            else: 
                break
        return day_to_use
    
    
    @staticmethod
    def get_masks_from_LC(LC_data, N_classes=17):
        masks = np.eye(N_classes)[(LC_data-1).astype(np.uint8)]
        return np.transpose(masks, [2, 0, 1])
    
    
    def get_static_features(self, reg, year, day, cell_bounds):
        path_to_static = f'{self.dataset_path}/{reg}/static/topo.npy'

        tmp_path = self.dataset_path

        path_to_temporary_year = f'{tmp_path}/{reg}/temporary_year/y{year}.npy'
        
        day_8_to_use = NPYdataset.get_availible_modis_day(day, self.days_8_list)
        day_16_to_use = NPYdataset.get_availible_modis_day(day, self.days_16_list)

        year_tmp = year
            
        path_to_temporary_8_days = f'{tmp_path}/{reg}/temporary_8_days/{year_tmp}/d{int(day_8_to_use):03d}.npy'
        path_to_temporary_16_days = f'{tmp_path}/{reg}/temporary_16_days/{year_tmp}/d{int(day_16_to_use):03d}.npy'

        if self.weather_data == "era5":
            paths = [path_to_static, path_to_temporary_year,
                     path_to_temporary_8_days, path_to_temporary_16_days]
        
        static = []
        for path in paths:
            if path is path_to_temporary_year:
                data = np.load(path, mmap_mode='r')[:, cell_bounds.top:cell_bounds.bottom+1,
                                                    cell_bounds.left:cell_bounds.right+1]
                static.append(data[:1]) # add population density
                    
                if self.LC_mode == 'binary_vector':
                    static.append(NPYdataset.get_masks_from_LC(data[1]))
                elif self.LC_mode == 'float':
                    static.append(data[1:2]/17) # use slicing to preserve 3 dims
                elif self.LC_mode == 'drop':
                    pass
                
            else:
                data = np.load(path, mmap_mode='r')[:, cell_bounds.top:cell_bounds.bottom+1,
                                                    cell_bounds.left:cell_bounds.right+1]
                static.append(data)

        static = np.concatenate(static, axis=0)
        return static
        
    
    def get_cell_features(self, reg, year, day, seq_length, cell_bounds, padding_bounds): 
        dynamic = self.get_dynamic_features(reg, year, day, seq_length, cell_bounds)
        #print("dynamic.shape:", dynamic.shape)
        dynamic = torch.tensor(dynamic, dtype=torch.float32)
        dynamic = pad(dynamic, padding_bounds, mode='reflect')

        static = self.get_static_features(reg, year, day, cell_bounds)
        #print("static.shape:", static.shape)
        static = torch.tensor(static, dtype=torch.float32)
        static = pad(static, padding_bounds, mode='reflect')
        
        return static, dynamic


    def __getitem__(self, idx):
        if self.mode == 'test_features':
            return self.get_test_features_item(idx)
        else:
            return self.getitem(idx)    

    
    def get_test_features_item(self, idx):
        idx = idx % self.len
        sample = self.samples_table.iloc[idx]

        year = sample.target_date.year
        day = sample.target_date.dayofyear
        reg = sample.reg
        self.upload_region_raster_coords(reg)
        target_day = int(sample.target_day)

        bounds = sample.geometry.bounds
        center = ((bounds[1]+bounds[3])/2, (bounds[2]+bounds[0])/2)
        cell_bounds, padding_bounds = NPYdataset.get_raster_coords_extended(
            self.raster_coords[reg]['latitude'],
            self.raster_coords[reg]['longitude'],
            center,
            self.sample_raster_size
        )
        
        static, dynamic = self.get_cell_features(
            reg, year, day, seq_length=17,
            cell_bounds=cell_bounds,
            padding_bounds=padding_bounds
        )

        day = torch.tensor([day, ])  
        return dynamic, day

    
    
    def getitem(self, idx):
        idx = idx % self.len
        sample = self.samples_table.iloc[idx]

        year = sample.target_date.year
        day = sample.target_date.dayofyear
        reg = sample.reg
        
        self.upload_region_raster_coords(reg)
        target_day = int(sample.target_day)

        bounds = sample.geometry.bounds
        center = [(bounds[1]+bounds[3])/2, (bounds[2]+bounds[0])/2]
        cell_bounds, padding_bounds = NPYdataset.get_raster_coords_extended(
            self.raster_coords[reg]['latitude'],
            self.raster_coords[reg]['longitude'],
            center,
            self.sample_raster_size
        )
        
        static, dynamic = self.get_cell_features(
            reg, year, day, seq_length=7,
            cell_bounds=cell_bounds,
            padding_bounds=padding_bounds
        )
        
        if self.augmenter:
            static_shape = static.shape
            dynamic_shape = dynamic.shape
            
            dynamic_tmp = dynamic.reshape(dynamic_shape[0] * dynamic_shape[1], *dynamic_shape[2:])
            cat_features = torch.cat([static, dynamic_tmp], axis=0)
            cat_features = self.augmenter(cat_features)
            static = cat_features[:static_shape[0]]
            dynamic = cat_features[static_shape[0]:].reshape(dynamic_shape)

        
        if self.mode in ['binary', 'binary_first_day']:
            targets = torch.zeros((1, ))          
            if target_day>0:
                targets += 1
        elif self.mode == 'fire_day_classification':
            targets = torch.zeros((5, ))
            targets[target_day-1] = 1.
        elif self.mode == 'days_group':
            if target_day in (1, 2, 3):    
                targets = torch.tensor((0,), dtype=torch.float32)
            if target_day in (4, 5):    
                targets = torch.tensor((1,), dtype=torch.float32)
        elif self.mode == 'days':
            targets = torch.tensor((target_day,), dtype=torch.float32)
        elif self.mode in ['multy_days', 'only_fires']:
            targets = torch.zeros((5, ))
            if target_day>0:
                targets[target_day - 1:] = 1.

        if self.augmenter:
            day_rand_wing = 5
            center_rand_mean = 0.2
            day += torch.randint(low=-day_rand_wing, high=day_rand_wing+1, size=(1,))
            center[0] += center_rand_mean*torch.randn(1)
            center[1] += center_rand_mean*torch.randn(1)
        
        if self.sample_with_date and self.sample_with_coords:
            return static, dynamic, torch.tensor([day, center[0], center[1]]), targets
        elif self.sample_with_date:
            return static, dynamic, torch.tensor([day]), targets
        elif self.sample_with_coords:
            return static, dynamic, torch.tensor([center[0], center[1]]), targets
        else:
            return static, dynamic, targets

