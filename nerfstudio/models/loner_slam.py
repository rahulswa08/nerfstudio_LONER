# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
NeRF implementation that combines many recent advancements.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Tuple, Type

import numpy as np
import torch
from torch.nn import Parameter
from torchmetrics.functional import structural_similarity_index_measure
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from nerfstudio.cameras.rays import RayBundle, RaySamples
from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackAttributes, TrainingCallbackLocation
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.field_components.spatial_distortions import SceneContraction
from nerfstudio.fields.density_fields import HashMLPDensityField
from nerfstudio.fields.nerfacto_field import NerfactoField
from nerfstudio.model_components.losses import (
    MSELoss,
    distortion_loss,
    interlevel_loss,
    orientation_loss,
    pred_normal_loss,
    scale_gradients_by_distance_squared,
)
from nerfstudio.model_components.ray_samplers import ProposalNetworkSampler, UniformSampler
from nerfstudio.model_components.renderers import AccumulationRenderer, DepthRenderer, NormalsRenderer, RGBRenderer
from nerfstudio.model_components.scene_colliders import NearFarCollider
from nerfstudio.model_components.shaders import NormalsShader
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.utils import colormaps
from nerfstudio.configs.config_utils import to_immutable_dict

import argparse
import os
import pathlib
import pickle
import re
import sys
import time
import torch
import tqdm
import pandas as pd
from scipy.spatial.transform import Rotation, Slerp
from scipy.interpolate import interp1d
import numpy as np
import torch
import torch.multiprocessing as mp
from torch.profiler import profile, record_function, ProfilerActivity
from scipy.spatial.transform import Rotation as R


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(PROJECT_ROOT)
sys.path.append(PROJECT_ROOT + "/src")

IM_SCALE_FACTOR = 1

from nerfstudio.cloner_slam_dir.analysis.render_utils import *

from nerfstudio.cloner_slam_dir.src.common.pose import Pose
from nerfstudio.cloner_slam_dir.src.common.pose_utils import WorldCube
from nerfstudio.cloner_slam_dir.src.common.ray_utils import CameraRayDirections
from nerfstudio.cloner_slam_dir.src.models.losses import *
from nerfstudio.cloner_slam_dir.src.models.model_tcnn import Model as ClonerModel
from nerfstudio.cloner_slam_dir.src.models.model_tcnn import OccupancyGridModel
from nerfstudio.cloner_slam_dir.src.models.ray_sampling import UniformRaySampler, OccGridRaySampler
from nerfstudio.cloner_slam_dir.analysis.utils import *


@dataclass
class ClonerSlamModelConfig(ModelConfig):
    """Nerfacto Model Config"""

    _target: Type = field(default_factory=lambda: ClonerSlamModel)

    experiment_directory: str = "~/LONER_SLAM/outputs"
    debug: bool = False
    eval: bool = False
    ckpt_id: str = ""
    use_gt_poses: bool = False
    use_raw_gt_poses: bool = False
    no_render_stills: bool = False
    render_video: bool = False
    no_interp: bool = False
    skip_step: bool = False
    only_last_frame: bool = False
    sep_ckpt_result: bool = False
    start_frame: int = 0
    traj: str = ""
    use_est_traj: bool = False
    render_global: bool = False
    # can remove these ig
    enable_collider: bool = True
    collider_params: Dict[str, float] = to_immutable_dict({"near_plane": 2.0, "far_plane": 6.0})
    loss_coefficients: Dict[str, float] = to_immutable_dict({"rgb_loss_coarse": 1.0, "rgb_loss_fine": 1.0})
    eval_num_rays_per_chunk: int = 4096
    prompt: str = ""


class ClonerSlamModel(Model):
    """Nerfacto model

    Args:
        config: Nerfacto configuration to instantiate model
    """

    config: ClonerSlamModelConfig

    def populate_modules(self):
        """Set the fields and modules."""
        super().populate_modules()

        checkpoints = os.listdir(f"{self.config.experiment_directory}/checkpoints")
        checkpoint = ""
        self.CHUNK_SIZE = 1024
        if self.config.ckpt_id is None:
            if "final.tar" in checkpoints:
                self.config.checkpoint = "final.tar"
            else:
                # https://stackoverflow.com/a/2669120
                convert = lambda text: int(text) if text.isdigit() else text
                alphanum_key = lambda key: [convert(c) for c in re.split("([0-9]+)", key)]
                checkpoint = sorted(self.config.checkpoints, key=alphanum_key)[-1]
        else:
            checkpoint = f"ckpt_{self.config.ckpt_id}.tar"

        checkpoint_path = pathlib.Path(f"{self.config.experiment_directory}/checkpoints/{checkpoint}")

        if self.config.sep_ckpt_result:
            render_dir = pathlib.Path(
                f"{self.config.experiment_directory}/renders/{checkpoint}_start{self.config.start_frame}_step{self.config.skip_step}"
            )
        else:
            render_dir = pathlib.Path(f"{self.config.experiment_directory}/renders")
        os.makedirs(render_dir, exist_ok=True)

        # override any params loaded from yaml
        with open(f"{self.config.experiment_directory}/full_config.pkl", "rb") as f:
            full_config = pickle.load(f)

        if full_config["calibration"]["camera_intrinsic"]["width"] is not None:
            full_config["calibration"]["camera_intrinsic"]["width"] *= IM_SCALE_FACTOR
            full_config["calibration"]["camera_intrinsic"]["height"] *= IM_SCALE_FACTOR
            full_config["calibration"]["camera_intrinsic"]["k"] *= IM_SCALE_FACTOR
            full_config["calibration"]["camera_intrinsic"]["new_k"] *= IM_SCALE_FACTOR
        else:
            # Sensible defaults for lidar-only case
            full_config["calibration"]["camera_intrinsic"]["width"] = int(1024 / 2 * IM_SCALE_FACTOR)
            full_config["calibration"]["camera_intrinsic"]["height"] = int(768 / 2 * IM_SCALE_FACTOR)
            full_config["calibration"]["camera_intrinsic"]["k"] = (
                torch.Tensor([[302, 0.0, 260], [0.0, 302, 197], [0.0, 0.0, 1.0]]) * IM_SCALE_FACTOR
            )
            full_config["calibration"]["camera_intrinsic"]["new_k"] = full_config["calibration"]["camera_intrinsic"][
                "k"
            ]
            full_config["calibration"]["camera_intrinsic"]["distortion"] = torch.zeros(4)
            full_config["calibration"]["lidar_to_camera"]["orientation"] = np.array(
                [0.5, -0.5, 0.5, -0.5]
            )  # for weird compatability

        intrinsic = full_config.calibration.camera_intrinsic
        self.im_size = torch.Tensor([intrinsic.height, intrinsic.width])

        if self.config.debug:
            full_config["debug"] = True

        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

        self._DEVICE = torch.device(full_config.mapper.device)

        torch.set_default_tensor_type("torch.cuda.FloatTensor")

        if not checkpoint_path.exists():
            print(f"Checkpoint {checkpoint_path} does not exist. Quitting.")
            exit()

        self.scale_factor = full_config.world_cube.scale_factor.to(self._DEVICE)
        shift = full_config.world_cube.shift
        self.world_cube = WorldCube(self.scale_factor, shift).to(self._DEVICE)

        self.ray_directions = CameraRayDirections(
            full_config.calibration, chunk_size=self.CHUNK_SIZE, device=self._DEVICE
        )

        # use single fine MLP when using OGM
        model_config = full_config.mapper.optimizer.model_config.model
        self.model = ClonerModel(model_config).to(self._DEVICE)

        print(f"Loading checkpoint from: {checkpoint_path}")
        ckpt = torch.load(str(checkpoint_path))
        self.model.load_state_dict(ckpt["network_state_dict"])

        if full_config.mapper.optimizer.samples_selection.strategy == "OGM":
            occ_model_config = full_config.mapper.optimizer.model_config.model.occ_model
            assert isinstance(occ_model_config, dict), f"OGM enabled but model.occ_model is empty"
            # Returns the 3D logits as a 5D tensor
            occ_model = OccupancyGridModel(occ_model_config).to(self._DEVICE)
            self.ray_sampler = OccGridRaySampler()
            occ_model.load_state_dict(ckpt["occ_model_state_dict"])
            # initialize occ_sigma
            occupancy_grid = occ_model()
            self.ray_sampler.update_occ_grid(occupancy_grid.detach())
        else:
            self.ray_sampler = UniformRaySampler()

        cfg = full_config.mapper.optimizer.model_config
        self.ray_range = cfg.data.ray_range

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = {}
        param_groups["proposal_networks"] = list(self.proposal_networks.parameters())
        param_groups["fields"] = list(self.field.parameters())
        return param_groups

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        callbacks = []
        return callbacks

    def get_outputs(self, ray_bundle: RayBundle):
        
        tic_img = time.time()
        size = (int(self.im_size[0]), int(self.im_size[1]), 1)
        rgb_size = (int(self.im_size[0]), int(self.im_size[1]), 3)
        rgb_fine = torch.zeros(rgb_size, dtype=torch.float32).view(-1, 3)
        depth_fine = torch.zeros(size, dtype=torch.float32).view(-1, 1)
        peak_depth_consistency = torch.zeros(size, dtype=torch.float32).view(-1, 1)
        print("--------------------")
        print("render_dataset_frame")

        for chunk_idx in range(self.ray_directions.num_chunks):
            
            eval_rays = ray_bundle
            eval_rays = eval_rays.to(self._DEVICE)

            results = self.model(eval_rays, self.ray_sampler, self.scale_factor, testing=True)

            rgb_fine[chunk_idx * self.CHUNK_SIZE : (chunk_idx + 1) * self.CHUNK_SIZE, :] = results["rgb_fine"]
            depth = results["depth_fine"].unsqueeze(1)
            depth_fine[chunk_idx * self.CHUNK_SIZE : (chunk_idx + 1) * self.CHUNK_SIZE, :] = results[
                "depth_fine"
            ].unsqueeze(1)

            s_vals = results["samples_fine"]
            weights_pred = results["weights_fine"]
            s_peaks = s_vals[torch.arange(eval_rays.shape[0]), weights_pred.argmax(dim=1)].unsqueeze(1)
            peak_depth_consistency[chunk_idx * self.CHUNK_SIZE : (chunk_idx + 1) * self.CHUNK_SIZE, :] = torch.abs(
                s_peaks - depth
            )

        rgb_fine = rgb_fine.reshape(1, rgb_size[0], rgb_size[1], rgb_size[2]).permute(0, 3, 1, 2)
        depth_fine = depth_fine.reshape(1, size[0], size[1], 1).permute(0, 3, 1, 2) * self.scale_factor
        peak_depth_consistency = (
            peak_depth_consistency.reshape(1, size[0], size[1], 1).permute(0, 3, 1, 2) * self.scale_factor
        )
        print(f"Took: {time.time() - tic_img} seconds for rendering an image")

        outputs = {
            "rgb": rgb_fine.clamp(0, 1),
            "accumulation": peak_depth_consistency,
            "depth": depth_fine,
        }

        return outputs

    