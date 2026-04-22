from torchvision import models
import torch


##### the following two sentence can directly load the pretrained model
# model = Resnet_model_original.resnet101(pretrained=True)
# resnet = models.resnet101(pretrained=True)

def load_pretrained_encoder_parameter_to_CTmodel(model,path):
    # load the checkpoin
    checkpoint = torch.load(path)

    # load the dictionary
    pretrained_dict = checkpoint['state_dict']
    model_dict = model.state_dict()

    # log the same name layer parameter
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # print(pretrained_dict.keys())
    model_dict.update(pretrained_dict)

    # load the modified model parameter
    model.load_state_dict(model_dict)

    return model

def load_pretrained_encoder_parameter(model,path):
    # load the checkpoin
    checkpoint = torch.load(path)

    # load the dictionary
    pretrained_dict = checkpoint['state_dict']
    model_dict = model.visual.state_dict()

    # log the same name layer parameter
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # print(pretrained_dict.keys())
    model_dict.update(pretrained_dict)

    # load the modified model parameter
    model.visual.load_state_dict(model_dict)

    return model



def product_the_pretrained_resnet_model(model):
    """
    #####################################################
    # contrast_model: the part layer pretrained model
    ####################################################
    """

    resnet_50 = models.resnet50(pretrained=True)

    # log the parameter of the pretrained model
    pretrained_dict = resnet_50.state_dict()
    model_dict = model.state_dict()

    # log the same name layer parameter
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # print(pretrained_dict.keys())
    model_dict.update(pretrained_dict)

    # load the modified model parameter
    model.load_state_dict(model_dict)   # the output size is not same  wo change the last two layer's name

    return model

def product_the_pretrained_densenet_model(model):
    """
    #####################################################
    # contrast_model: the part layer pretrained model
    ####################################################
    """

    densenet121 = models.densenet121(pretrained=True)

    # log the parameter of the pretrained model
    pretrained_dict = densenet121.state_dict()
    model_dict = model.state_dict()

    # log the same name layer parameter
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # print(pretrained_dict.keys())
    model_dict.update(pretrained_dict)

    # load the modified model parameter
    model.load_state_dict(model_dict)   # the output size is not same  wo change the last two layer's name

    return model

def product_the_pretrained_VIT_model(model):
    """
    #####################################################
    # contrast_model: the part layer pretrained model
    ####################################################
    """
    from timm import models
    vit_tiny_patch16_224 = models.vit_tiny_patch16_224(pretrained=True)

    # log the parameter of the pretrained model
    pretrained_dict = vit_tiny_patch16_224.state_dict()
    model_dict = model.state_dict()

    # log the same name layer parameter
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # print(pretrained_dict.keys())
    model_dict.update(pretrained_dict)

    # load the modified model parameter
    model.load_state_dict(model_dict)   # the output size is not same  wo change the last two layer's name

    return model

def product_the_pretrained_swin_transformer_model(model):
    """
    #####################################################
    # contrast_model: the part layer pretrained model
    ####################################################
    """
    from timm import models
    swin_tiny_patch4_window7_224 = models.swin_tiny_patch4_window7_224(pretrained=True)

    # log the parameter of the pretrained model
    pretrained_dict = swin_tiny_patch4_window7_224.state_dict()
    model_dict = model.state_dict()

    # log the same name layer parameter
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # print(pretrained_dict.keys())
    model_dict.update(pretrained_dict)

    # load the modified model parameter
    model.load_state_dict(model_dict)   # the output size is not same  wo change the last two layer's name

    return model

def freeze_target_layer(contrast_model):
    """
    #####################################################
    # contrast_model: the model
    # optim_model_parameters: the parameter of the optim
    ####################################################
    """
    for name, param in contrast_model.named_parameters():
        # print(name)
        if "features.denseblock2.denselayer14"  in name:
            param.requires_grad = True
        elif "features.denseblock2.denselayer5"  in name:
            param.requires_grad = True
        elif "features.norm_modified" in name:
            param.requires_grad = True
        elif "classifier_modified" in name:
            param.requires_grad = True
        elif "transition.norm" in name:
            param.requires_grad = True
        elif "transition.conv" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    for name, param in contrast_model.named_parameters():
        if param.requires_grad:
            print("requires_grad: True ", name)
        else:
            print("requires_grad: False ", name)


    opti_model_parameters = filter(lambda p: p.requires_grad, contrast_model.parameters())
    return opti_model_parameters