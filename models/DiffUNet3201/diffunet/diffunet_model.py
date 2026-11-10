from .nnunet2d_denoise import get_nnunet2d_denoise
from .nnunet2d import get_nnunet2d
import torch.nn as nn 

class DiffUNet(nn.Module):
    def __init__(self, in_channels, out_channels, 
                 ddim_steps=3, rand_steps=1, bta=True, enable_sdf=False,
                 enable_gcf=False, gcf_stage=2, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if enable_gcf and (not enable_sdf or out_channels != 2):
            raise ValueError("GCF requires enable_sdf=True and binary segmentation")
        self.enable_gcf = enable_gcf

        self.edge_model = get_nnunet2d(in_chans=in_channels, out_chans=out_channels,
                                       enable_sdf=enable_sdf)
        self.denoise_model = get_nnunet2d_denoise(in_chans=in_channels, out_chans=out_channels, 
                                          ddim_step=ddim_steps,
                                          rand_step=rand_steps,
                                          bta=bta,
                                          enable_gcf=enable_gcf,
                                          gcf_stage=gcf_stage)

    def forward(self, image, gt=None, ddim=False, return_sdf=False):
        if return_sdf or self.enable_gcf:
            pred_edge, embeddings, pred_sdf = self.edge_model(image, return_sdf=True)
        else:
            pred_edge, embeddings = self.edge_model(image)

        if ddim:
            pred = self.denoise_model(image, gt=gt, 
                                        embeddings=embeddings, 
                                        ddim=True,
                                        sdf=pred_sdf if self.enable_gcf else None)
            output = pred + pred_edge
            return (output, pred_sdf) if return_sdf else output
        else :
            pred, uncertainty = self.denoise_model(image, gt=gt, 
                                        embeddings=embeddings, 
                                        ddim=False,
                                        sdf=pred_sdf if self.enable_gcf else None)
            if return_sdf:
                return pred, pred_edge, uncertainty, pred_sdf
            return pred, pred_edge, uncertainty
    '''
    def forward(self, image, gt=None, ddim=False):
        pred_edge, embeddings = self.edge_model(image)

        pred = self.denoise_model(image, gt=gt, 
                                        embeddings=embeddings, 
                                        ddim=True)
        return pred + pred_edge
    '''
