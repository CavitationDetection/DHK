import torch
import timm

def vit_tiny_patch16_224(num_classes: int,**kwargs):
    model = timm.create_model("vit_tiny_patch16_224", pretrained = False)
    in_features = model.head.in_features
    model.head = torch.nn.Linear(in_features, num_classes)
    return model

def vit_small_patch16_224(num_classes: int,**kwargs):
    model = timm.create_model("vit_small_patch16_224", pretrained = False)
    in_features = model.head.in_features
    model.head = torch.nn.Linear(in_features, num_classes)
    return model

def vit_small_patch32_224(num_classes: int,**kwargs):
    model = timm.create_model("vit_small_patch32_224", pretrained = False)
    in_features = model.head.in_features
    model.head = torch.nn.Linear(in_features, num_classes)
    return model

def vit_base_patch32_224(num_classes: int,**kwargs):
    model = timm.create_model("vit_base_patch32_224", pretrained = False)
    in_features = model.head.in_features
    model.head = torch.nn.Linear(in_features, num_classes)
    return model

def vit_base_patch16_224(num_classes: int, **kwargs):
    model = timm.create_model("vit_base_patch16_224", pretrained = False)
    in_features = model.head.in_features
    model.head = torch.nn.Linear(in_features, num_classes)
    return model