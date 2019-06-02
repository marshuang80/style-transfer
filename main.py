import numpy as np
import argparse
import torch
import torch.nn as nn
import torchvision.models as models
import matplotlib.pyplot as plt
import scipy.misc as misc
import copy
from loss import ContentLoss, StyleLoss
import logger


torch.autograd.set_detect_anomaly(True)

def build_model(pretrained, style, content, content_layer=12):

    model = nn.Sequential()
    count = 0

    for idx, layer in enumerate(pretrained.children()):
        if isinstance(layer, nn.Conv2d):
            model.add_module(f"Conv2D_{count}", layer)
            style_output = model(style).detach()
            model.add_module(f"Loss_{count}", StyleLoss(style_output))
            count += 1
            
        else:
            model.add_module(f"{idx}", layer)

        if count == content_layer:
            content_output = model(content).detach()
            model.add_module(f"Content_loss", ContentLoss(content_output))

    return model


def main(args):

    # tensorboard 
    logger_tb = logger.Logger(log_dir=args.experiment_name)


    # load pretrained model    
    vgg16 = models.vgg16(pretrained=True).features.eval()

    # load in images
    style = plt.imread(args.style)
    content = plt.imread(args.content)
    content = misc.imresize(content, style.shape)

    # normalize TODO normalize for channel
    norm = lambda x: (x - x.mean()) / x.std()
    content = norm(content)
    style = norm(style)

    # create white noise image 
    input_img = np.random.normal(0, 1, size=(style.shape))

    # to tensor
    h,w,c = style.shape
    content = torch.Tensor(content).reshape(1,c,h,w)
    style = torch.Tensor(style).reshape(1,c,h,w)
    input_img = torch.from_numpy(input_img).reshape(1,c,h,w).float()

    #vgg16.to(args.device)
    model = build_model(vgg16, style, content)

    # weight ratio
    num_parts = 1/args.weight_ratio 
    a = 1 / (num_parts + 1)
    b = a * num_parts

    # optimizer 
    optimizer = torch.optim.LBFGS([input_img.requires_grad_()])

    # use cuda
    input_img.to(args.device)
    
    for epoch in range(args.epoch):

        def closure():

            optimizer.zero_grad()

            _ = model(input_img)

            # define loss as 0
            style_loss, content_loss = 0, 0
            loss_layer_count = 0
            
            # get loss from loss layers
            for layer in model.children():

                if isinstance(layer, StyleLoss):
                    style_loss += layer.loss
                    loss_layer_count += 1

                if isinstance(layer, ContentLoss):
                    content_loss += layer.loss

            style_loss /= loss_layer_count
            total_loss = a * content_loss + b * style_loss
            total_loss.backward()

            return total_loss

        optimizer.step(closure)
        logger_tb.update_loss('train_loss ', total_loss.item(), epoch)
        logger_tb.update_image('image ', input_img.squeeze().permute(2,1,0).cpu().detach().numpy(), epoch)



if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--style', type=str)
    parser.add_argument('--content', type=str)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--weight_ratio', type=float, default=1e-1)

    args = parser.parse_args()

    main(args)

