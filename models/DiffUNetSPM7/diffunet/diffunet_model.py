from .nnunet2d_denoise import get_nnunet2d_denoise
from .nnunet2d import get_nnunet2d
import torch.nn as nn 

class DiffUNet(nn.Module):
    def __init__(self, in_channels, out_channels, 
                 ddim_steps=3, rand_steps=1, bta=True,config=None, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.edge_model = get_nnunet2d(in_chans=in_channels, out_chans=out_channels,config=config)
        self.denoise_model = get_nnunet2d_denoise(in_chans=in_channels, out_chans=out_channels, 
                                          ddim_step=ddim_steps,
                                          rand_step=rand_steps,
                                          bta=bta)

    def forward(self, image, gt=None, ddim=False):
        pred_edge, embeddings = self.edge_model(image)
        #print("pred_edge",pred_edge.shape) 
        #(1,2,256,256)
        #for i, feat in enumerate(embeddings):
        #    print(f"[stage {i}] shape={feat.shape}")
        #(1,32,256,256)
        #(1,64,128,128)
        #(1,128,64,64)
        #(1,256,32,32)
        #(1,320,16,16)
        #(1,320,8,8)
        if ddim:
            pred = self.denoise_model(image, gt=gt, 
                                        embeddings=embeddings, 
                                        ddim=True)
            return pred + pred_edge
        else :
            pred, uncertainty = self.denoise_model(image, gt=gt, 
                                        embeddings=embeddings, 
                                        ddim=False)
            return pred, pred_edge, uncertainty
    '''
    def forward(self, image, gt=None, ddim=False):
        pred_edge, embeddings = self.edge_model(image)

        pred = self.denoise_model(image, gt=gt, 
                                        embeddings=embeddings, 
                                        ddim=True)
        return pred + pred_edge
    '''
