from typing import Tuple, Optional, Union, List
from dataclasses import dataclass
import torch
from lightning import LightningModule

from internal.density_controllers.vanilla_density_controller import VanillaDensityController, VanillaDensityControllerImpl

@dataclass
class EdgeDensityController(VanillaDensityController):
    densify_grad_threshold_at_edge: float = 20.0 # 图像边缘的梯度阈值

    def instantiate(self, *args, **kwargs) -> 'EdgeDensityControllerImpl':
        return EdgeDensityControllerImpl(self)


class EdgeDensityControllerImpl(VanillaDensityControllerImpl):
    config: EdgeDensityController