import numpy as np
import argparse
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision
import matplotlib.pyplot as plt
import copy
import utils
import logger
import tqdm
import scipy.misc
from loss import ContentLoss, StyleLoss

def main(args):

    # tensorboard 
    logger_tb = logger.Logger(log_dir=args.experiment_name)

    # use cuda
    device = torch.device(args.device) 

    # load pretrained model    
    vgg19 = models.vgg19(pretrained=True).features.eval()

    # load in images
    style, content, input_img = utils.parse_img(args)

    # display initial style and content image on tb
    logger_tb.update_image('style', style, 0)
    logger_tb.update_image('content', content, 0)

    # build model
    model = utils.build_model(vgg19, style, content, device)

    # weight ratio (a: content weight, b: style weight)
    a, b = utils.get_weight_ratio(args.weight_ratio)

    # use CUDA if availible
    input_img = input_img.to(device)
    model = model.to(device)

    # optimizer 
    optimizer = torch.optim.LBFGS([input_img.requires_grad_()])

    with tqdm.tqdm(total=len(range(args.epoch)), unit=f"epoch") as progress_bar:
        for epoch in range(args.epoch):

            # show example image on tensorboard
            updated_img = input_img.squeeze().cpu().detach().numpy()
            logger_tb.update_image('transformation ', updated_img, epoch)

            def closure():
                '''closure function for LBFGS optimizer'''

                # forward pass
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

                # back propagate loss
                style_loss /= loss_layer_count
                total_loss = a * content_loss + b * style_loss
                total_loss.backward()
                logger_tb.update_loss('train_loss ', total_loss.item(), epoch)
                return total_loss
            
            # TODO gets nan values error
            optimizer.step(closure)

            # prevent out of range pixels
            input_img.data.clamp_(0,1)

            progress_bar.update(1)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--style', type=str)
    parser.add_argument('--content', type=str)
    parser.add_argument('--size', type=int, default=None)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--weight_ratio', type=float, default=1e-7)
    parser.add_argument('--experiment_name', type=str, default='test')
    parser.add_argument('--init_white_noise', type=bool, default=False)
    parser.add_argument('--square_img', type=bool, default=True)

    args = parser.parse_args()

    main(args)

