import torch
import torch.nn as nn 

class StyleLoss(nn.Module):                                          
                                                                     
    def __init__(self, ori_style):                                   
        super(StyleLoss, self).__init__()
        self.ori_style = ori_style.squeeze()                         
        self.l, self.w, self.h = self.ori_style.shape                
        self.ori_style = self.ori_style.view(self.l, self.w*self.h)  
        self.loss = None                                             
                                                                     
    def forward(self, input_style):                                  
        input_style_out = input_style.squeeze().clone()
        input_style_out = input_style_out.view(self.l, self.w*self.h)
                                                                     
        # matrix multiplication                                      
        G = self.ori_style @ self.ori_style.transpose(0,1)
        A = input_style_out @ input_style_out.transpose(0,1)
                                                                     
        self.loss = torch.sum((G - A) **2 ) * 1/(4*(self.l**2)*(self.w*self.h)**2)  
                                                                     
        return input_style                                           

