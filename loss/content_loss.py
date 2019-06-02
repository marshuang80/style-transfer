import torch
import torch.nn as nn

class ContentLoss(nn.Module):                                                   
                                                                                
    def __init__(self, ori_content):                                            
        super(ContentLoss, self).__init__()
        ori_content = ori_content.squeeze()                                     
        self.l, self.w, self.h = ori_content.shape                              
        self.ori_content = ori_content.view(self.l, self.w*self.h)              
        self.loss = None                                                        
                                                                                
    def forward(self, input_content):                                           
                                                                                
        input_content_out = input_content.squeeze()                             
        input_content_out = input_content_out.view(self.l, self.w*self.h)       
        self.loss = torch.sum((self.ori_content - input_content_out) ** 2) * 0.5
                                                                                
        return  input_content                                                   

