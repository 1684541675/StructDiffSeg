from .nnunet2d_denoise import get_nnunet2d_denoise
from .nnunet2d import get_nnunet2d
import torch.nn as nn 
from ..spmnet.unet import network



class DiffUNet(nn.Module):
    def __init__(self, in_channels, out_channels, 
                 ddim_steps=3, rand_steps=1, bta=True,config=None, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.edge_model = network(in_channel=in_channels,out_channel=out_channels,config=config)
        self.denoise_model = get_nnunet2d_denoise(in_chans=in_channels, out_chans=out_channels, 
                                          ddim_step=ddim_steps,
                                          rand_step=rand_steps,
                                          bta=bta)

    def forward(self, image, gt=None, ddim=False):
        pred_edge, embeddings = self.edge_model(image)
        #print("pred_edge",pred_edge.shape) 
        #for i, feat in enumerate(embeddings):
        #    print(f"[stage {i}] shape={feat.shape}")
        
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
