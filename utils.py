import matplotlib.pyplot as plt 
import scipy.misc as misc
import numpy as np
import torch 
import torch.nn as nn
from loss import ContentLoss, StyleLoss

def parse_img(args):
    """parse input style and content image to tensor
    """

    # load in images                                                                  
    style = plt.imread(args.style)                                                    
    content = plt.imread(args.content)                                                
                                              
    # TODO so far only 128 images works well                                          
    if args.size is not None:                                                         

        # content should keep ratio so use as reference
        if args.square_img:
            content = misc.imresize(content, (args.size, args.size))
        else:
            h_ratio = args.size / float(content.shape[0])                                   
            h_size = int((float(content.shape[1]) * float(h_ratio)))                        
            content = misc.imresize(content, (args.size, h_size))                             

    style = misc.imresize(style, content.shape)                                     
                                             
    # normalize between 0 and 1 (negative values mess up color)
    norm = lambda x: (x - x.min(axis=(0,1))) / (x.max(axis=(0,1)) - x.min(axis=(0,1)))
    content = norm(content)                                                           
    style = norm(style)                                                               
    
    # transpose to tensor board shape
    content = np.transpose(content,(2,0,1))                                           
    style = np.transpose(style,(2,0,1))                                               
                                              
    # to tensor                                                                       
    content = torch.Tensor(content).unsqueeze(0)                                      
    style = torch.Tensor(style).unsqueeze(0)                                          

    # get initial input image
    if args.init_white_noise == True:                                                 
        input_img = np.random.normal(0, 1, size=style.shape)                          
        input_img = torch.from_numpy(input_img).float()                               
    else:                                                                             
        input_img = content.clone()                                                   
    
    return style, content, input_img


def get_weight_ratio(weight_ratio):
    """ get the style and content weight ratio

    Returns
    -------
        a: content weight
        b: style weight
    """

    num_parts = 1/weight_ratio
    a = 1 / (num_parts + 1)        
    b = a * num_parts              
    return a, b

class Normalize(nn.Module):

    def __init__(self,device):

         super(Normalize, self).__init__()

         # vgg mean and std
         self.mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1).to(device)
         self.std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1).to(device)

    def forward(self, img):

        return (img - self.mean) / self.std


def build_model(pretrained, style, content, device, content_layers=[4], style_layers=[0,1,2,3,4,5,6,7]):
    '''build style transfer model from pretrained model
    '''

    model = nn.Sequential()
    count = 0

    # use device
    style = style.to(device)
    content = content.to(device)
    pretrained = pretrained.to(device)

    # max cnn layer
    max_layer = max(max(content_layers),max(style_layers))

    # gets weird color pixels if not noralized                                
    model.add_module('normalize', Normalize(device))

    # loop through pretrained model and add loss 
    for idx, layer in enumerate(pretrained.children()):

        if isinstance(layer, nn.Conv2d):

            # return model if max layer reached
            if count > max_layer: continue

            model.add_module(f"Conv2D_{count}", layer)
            count += 1

            # add content loss
            if count in content_layers:
                content_output = model(content).detach()
                model.add_module(f"Content_loss", ContentLoss(content_output))

            # add style loss
            if count in style_layers:
                style_output = model(style).detach()
                model.add_module(f"Loss_{count}", StyleLoss(style_output))

        else:
            model.add_module(f"{idx}", layer)

    return model

