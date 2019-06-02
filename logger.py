from tensorboardX import SummaryWriter
from datetime import datetime
import os
import numpy as np

class Logger(object):

    def __init__(self, log_dir):

        # check if logger directory exist
        if not os.path.exists('logs'):
            os.mkdir('./logs')
        self.log_dir = os.path.join('./logs', log_dir + '_{}.log'.format(datetime.now().strftime('%b%d_%H%M')))
        self.summary_writer = SummaryWriter(log_dir=self.log_dir)
    
    def update_loss(self, stage, loss, step):
        """stage in [test, train]"""
        self.summary_writer.add_scalar('{}_loss'.format(stage), loss, step)
    
    def update_image(self, stage, img, step):

        self.summary_writer.add_image(stage, img, step) 
