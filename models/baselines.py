import torch
import torch.nn as nn
import _pickle as cPickle


# def get_median_max(features):
#     return (torch.median(torch.median(features, -1)[0], -1)[0],
#             torch.max(torch.max(features, -1)[0], -1)[0])

def get_median_max(features):
    s_mean = torch.median(features.flatten(start_dim=-2), dim=-1)[0]
    s_max = torch.max(features.flatten(start_dim=-2), dim=-1)[0]
    return s_mean, s_max

class BaseForestModel(nn.Module):
    def __init__(self, path_to_model):
        super().__init__()
        self.path_to_model = path_to_model
        with open(path_to_model, 'rb') as f:
            self.clf = cPickle.load(f)
        self.threshold = 0.5
    
    def forward(self, static, dynamic, additional):
        # input features to one tensor
        features = []
    
        s_mean, s_max = get_median_max(static)
        features.append(s_mean)
        features.append(s_max)
    
        d_mean, d_max = get_median_max(dynamic)
        features.append(d_mean.flatten(start_dim=-2))
        features.append(d_max.flatten(start_dim=-2))
        
        features=torch.cat(features, axis=1) 

        # return probabilities
        out = self.clf.predict_proba(features)
        return torch.tensor(out)