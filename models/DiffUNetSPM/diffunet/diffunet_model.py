from .nnunet2d_denoise import get_nnunet2d_denoise
from .nnunet2d import get_nnunet2d
import torch
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

    def forward(self, image, gt=None, ddim=False, debug_trajectory=False,
                ablation_mode=None, condition_mode=None, return_branches=False,
                spm_scale=1.0):
        if spm_scale != 1.0 and (not ddim or debug_trajectory or ablation_mode is not None
                                or condition_mode not in (None, 'normal')):
            raise ValueError("SPM scaling requires DDIM inference without other ablations/debug modes")
        if return_branches and (not ddim or debug_trajectory):
            raise ValueError("return_branches requires DDIM inference without trajectory debugging")
        if debug_trajectory and not ddim:
            raise ValueError("debug_trajectory is only supported for DDIM inference")
        if ablation_mode is not None and not ddim:
            raise ValueError("ablation_mode is only supported for DDIM inference")
        if condition_mode not in (None, 'normal', 'zero_denoiser_image',
                                  'shuffle_denoiser_image', 'zero_spm_condition'):
            raise ValueError(f'Unknown image condition mode: {condition_mode}')
        if condition_mode is not None and not ddim:
            raise ValueError("condition_mode is only supported for DDIM inference")
        if debug_trajectory:
            pred_edge, embeddings, refined_shape_prior = self.edge_model(image, return_shape_prior=True)
        else:
            pred_edge, embeddings = self.edge_model(image)
        #print("pred_edge",pred_edge.shape) 
        #for i, feat in enumerate(embeddings):
        #    print(f"[stage {i}] shape={feat.shape}")
        
        if ddim:
            denoiser_image = image
            denoiser_embeddings = embeddings
            if condition_mode == 'zero_denoiser_image':
                denoiser_image = torch.zeros_like(image)
            elif condition_mode == 'shuffle_denoiser_image':
                batch, channels, height, width = image.shape
                devices = [image.device.index] if image.is_cuda else []
                with torch.random.fork_rng(devices=devices):
                    permutations = torch.stack([
                        torch.randperm(height * width, device=image.device)
                        for _ in range(batch)
                    ])
                denoiser_image = image.flatten(2).gather(
                    2, permutations[:, None, :].expand(-1, channels, -1)
                ).reshape_as(image)
            elif condition_mode == 'zero_spm_condition':
                denoiser_embeddings = [torch.zeros_like(feature) for feature in embeddings]
            # Keep scale=1 as the exact original path; scale=0 matches zero_spm_condition.
            if spm_scale == 0.0:
                denoiser_embeddings = [torch.zeros_like(feature) for feature in embeddings]
            elif spm_scale != 1.0:
                denoiser_embeddings = [feature * spm_scale for feature in embeddings]
            pred = self.denoise_model(denoiser_image, gt=gt,
                                        embeddings=denoiser_embeddings,
                                        ddim=True, debug_trajectory=debug_trajectory,
                                        ablation_mode=ablation_mode)
            if debug_trajectory:
                diffusion_logits, step_logits = pred
                return {
                    'step_logits': step_logits,
                    'diffusion_logits': diffusion_logits,
                    'spm_logits': pred_edge,
                    'final_logits': diffusion_logits + pred_edge,
                    'refined_shape_prior': refined_shape_prior,
                }
            if return_branches:
                return {'spm': pred_edge, 'diffusion': pred, 'final': pred + pred_edge}
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
