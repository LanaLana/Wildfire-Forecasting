import torch
import torch.nn as nn
import torch.nn.functional as F

### NON-fuzzy metrics ## 
class IOU():
    def __init__(self, eps=1e-8):
        self.eps = eps

    def __call__(self, outputs, labels):
        assert outputs.shape==labels.shape
        outputs = outputs.byte()
        labels = labels.byte()
        
        intersection = (outputs & labels).float().sum()  
        union = (outputs | labels).float().sum()

        iou = (intersection + self.eps) / (union + self.eps)
        return iou   


class JAC_BINARY():
    def __init__(self, eps = 1e-8):
        self.eps = eps

    def __call__(self, outputs, labels):
        assert outputs.shape==labels.shape
        
        outputs = outputs.byte()  
        labels = labels.byte()
        
        TP = (outputs & labels).float().sum((1, 2, 3))  
        FN = ((1-outputs) & labels).float().sum()
        FP = (outputs & (1-labels)).float().sum()
        
        jac = (TP + self.eps) / (TP + FN + FP + self.eps)
        return jac
    

class DICE_BINARY():
    def __init__(self, eps=1e-8, inverse=False):
        self.eps = eps
        self.inverse = inverse

    def __call__(self, outputs, labels):
        assert outputs.shape==labels.shape
        outputs = outputs.byte()  
        labels = labels.byte()
        if self.inverse:
            outputs = 1-outputs
            labels = 1-labels
            
        TP = (outputs & labels).float().sum()  
        FN = ((1-outputs) & labels).float().sum()
        FP = (outputs & (1-labels)).float().sum()
        
        dice = (2*TP + self.eps) / (2*TP + FN + FP + self.eps)
        return dice


class F1_BINARY():
    def __init__(self, eps=1e-8, inverse=False, logit=False):
        self.dice = DICE_BINARY(inverse=inverse)
        self.logit = logit

    def __call__(self, outputs, labels):
        if self.logit:
            outputs = torch.sigmoid(outputs)
        return self.dice(outputs, labels)


class BETA_F1_BINARY():
    def __init__(self, eps=1e-8, beta=1, inverse=False, bidirectional=False):
        self.beta = beta
        self.eps = eps
        self.inverse = inverse
        self.bidirectional = bidirectional

    def __call__(self, outputs, labels):
        assert outputs.shape==labels.shape
        outputs = outputs.byte()  
        labels = labels.byte()


        def calc(outputs, labels):
            TP = (outputs & labels).float().sum()  
            FN = ((1-outputs) & labels).float().sum()
            FP = (outputs & (1-labels)).float().sum()
            out = ((1+self.beta**2)*TP + self.eps) / ((1+self.beta**2)*TP + (self.beta**2) * FN + FP + self.eps)     
            return out

        if not self.bidirectional:
            if self.inverse:
                outputs = 1-outputs
                labels = 1-labels

            return calc(outputs, labels)

        else:
            out1 =  calc(outputs, labels)
            self.beta = 1. / self.beta
            out2 =  calc(1-outputs, 1-labels)
            return 0.5*(out1 + out2)


class SENSIVITY_BINARY():
    def __init__(self, eps = 1e-8, inverse=False):
        self.eps = eps
        self.inverse = inverse

    def __call__(self, outputs, labels):
        assert outputs.shape==labels.shape
        outputs = outputs.byte()  
        labels = labels.byte()
        if self.inverse:
            outputs = 1-outputs
            labels = 1-labels
        
        TP = (outputs & labels).float().sum()  
        FN = ((1-outputs) & labels).float().sum()
        
        sn = (TP + self.eps) / (TP + FN + self.eps)
        return sn
    

class SPECIFICITY_BINARY():
    def __init__(self, eps = 1e-8, inverse=False):
        self.eps = eps
        self.inverse = inverse

    def __call__(self, outputs, labels):
        assert outputs.shape==labels.shape
        outputs = outputs.byte() 
        labels = labels.byte()
        if self.inverse:
            outputs = 1-outputs
            labels = 1-labels
        
        TN = ((1-outputs) & (1-labels)).float().sum()  
        FP = (outputs & (1-labels)).float().sum()
        
        sp = (TN + self.eps) / (TN + FP + self.eps)
        return sp
    

class PRECISION_BINARY():
    def __init__(self, eps = 1e-8, inverse=False):
        self.eps = eps
        self.inverse = inverse
    
    def __call__(self, outputs, labels):
        assert outputs.shape==labels.shape
        outputs = outputs.byte()  
        labels = labels.byte()
        if self.inverse:
            outputs = 1-outputs
            labels = 1-labels
        
        TP = (outputs & labels).float().sum()  
        FP = (outputs & (1-labels)).float().sum()
        
        pr = (TP + self.eps) / (TP + FP + self.eps)
        return pr


class RECALL_BINARY():
    def __init__(self, eps = 1e-8, inverse=False, logit=False):
        self.eps = eps
        self.sn = SENSIVITY_BINARY(inverse=inverse)

    def __call__(self, outputs, labels):
        return self.sn(outputs, labels)


class MEAN_BIDIRECTIONAL_F1():
    def __init__(self, eps = 1e-8):
        self.F1_BINARY = F1_BINARY(inverse=False)
        self.F1_BINARY_rev = F1_BINARY(inverse=True)

    def __call__(self, outputs, labels):
        return (self.F1_BINARY(outputs, labels) + self.F1_BINARY_rev(outputs, labels))/2

