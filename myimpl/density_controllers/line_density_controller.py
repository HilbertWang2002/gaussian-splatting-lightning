import os
import numpy as np
from typing import Tuple, Optional, Union, List
from dataclasses import dataclass
import torch
from lightning import LightningModule
import limap.util.io as limapio
from internal.density_controllers.vanilla_density_controller import VanillaDensityController, VanillaDensityControllerImpl
from internal.density_controllers.density_controller import Utils
from internal.models.vanilla_gaussian import VanillaGaussianModel
from fast_point_line_distance import find_shortest_distance

@dataclass
class LineDensityController(VanillaDensityController):
    densify_strength_near_lines: float = 100
    lines_path: str = "lines/finaltracks"
    point_line_distance_threhold: float = 0.05

    def instantiate(self, *args, **kwargs) -> 'LineDensityControllerImpl':
        return LineDensityControllerImpl(self)


class LineDensityControllerImpl(VanillaDensityControllerImpl):
    config: LineDensityController

    def setup(self, stage: str, pl_module: LightningModule) -> None:
        super().setup(stage, pl_module)
        
        if stage=='fit':
            # read lines
            lines, linetracks = limapio.read_lines_from_input(os.path.join(pl_module.trainer.datamodule.hparams['path'], self.config.lines_path))
            self.lines_data = np.array([line.as_array() for line in lines])

            # initialize near line mask
            self.near_line_mask = self.compute_near_line_mask(pl_module.trainer.model.gaussian_model.get_xyz.detach().cpu().numpy(), pl_module)
        else:
            self.near_line_mask = None
    
    def after_backward(self, outputs: dict, batch, gaussian_model: VanillaGaussianModel, optimizers: List, global_step: int, pl_module: LightningModule) -> None:
        if global_step >= self.config.densify_until_iter:
            return
        pl_module.trainer.model.log('line_density_controller/near_line_mask', self.near_line_sum, prog_bar=True, on_step=True, on_epoch=False)
        with torch.no_grad():
            self.update_states(outputs)

            # densify and pruning
            if global_step > self.config.densify_from_iter and global_step % self.config.densification_interval == 0:
                size_threshold = 20 if global_step > self.config.opacity_reset_interval else None
                self._densify_and_prune(
                    max_screen_size=size_threshold,
                    gaussian_model=gaussian_model,
                    optimizers=optimizers,
                )
                # recompute near line mask
                self.near_line_mask = self.compute_near_line_mask(gaussian_model.get_xyz.detach().cpu().numpy(), pl_module)

            if global_step % self.config.opacity_reset_interval == 0 or \
                    (
                        torch.all(pl_module.background_color == 1.) and global_step == self.config.densify_from_iter
                    ):
                self._reset_opacities(gaussian_model, optimizers)
                self.opacity_reset_at = global_step
    
    def compute_near_line_mask(self, points, pl_module: LightningModule) -> np.ndarray:
        min_dist = find_shortest_distance(
            points,
            self.lines_data,
        )
        results = np.where(min_dist < self.config.point_line_distance_threhold, True, False)
        self.near_line_sum = results.sum()
        return results
    
    def _add_densification_stats(self, grad, update_filter, scale: Union[float, int, None]):
        scaled_grad = grad.clone()
        scaled_grad[self.near_line_mask] = scaled_grad[self.near_line_mask]*self.config.densify_strength_near_lines

        scaled_grad = grad[update_filter, :2]
        if scale is not None:
            scaled_grad = scaled_grad * scale
        grad_norm = torch.norm(scaled_grad, dim=-1, keepdim=True)

        self.xyz_gradient_accum[update_filter] += grad_norm
        self.denom[update_filter] += 1
        