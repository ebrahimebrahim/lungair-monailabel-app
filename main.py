# Copyright 2020 - 2021 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
from distutils.util import strtobool
from typing import Dict

from lib import MyInfer, MyStrategy, MyTrain
import monai

from monailabel.interfaces.app import MONAILabelApp
from monailabel.interfaces.tasks.infer import InferTask
from monailabel.interfaces.tasks.scoring import ScoringMethod
from monailabel.interfaces.tasks.strategy import Strategy
from monailabel.interfaces.tasks.train import TrainTask
from monailabel.scribbles.infer import HistogramBasedGraphCut
from monailabel.tasks.activelearning.epistemic import Epistemic
from monailabel.tasks.activelearning.random import Random
from monailabel.tasks.activelearning.tta import TTA
from monailabel.tasks.scoring.epistemic import EpistemicScoring
from monailabel.tasks.scoring.tta import TTAScoring

logger = logging.getLogger(__name__)


class MyApp(MONAILabelApp):
    def __init__(self, app_dir, studies, conf):
        spatial_dims = 2
        image_channels = 1
        seg_channels = 2; # lung, background
        seg_net_channel_seq = (8,16,32,32,32,64,64,64)
        stride_seq = (2,2,2,2,1,2,1)
        dropout_seg_net = 0.5
        num_res_units = 2

        self.seg_net = monai.networks.nets.UNet(
            spatial_dims = spatial_dims,
            in_channels = image_channels,
            out_channels = seg_channels,
            channels = seg_net_channel_seq,
            strides = stride_seq,
            dropout = dropout_seg_net,
            num_res_units = num_res_units
        )
        num_params = sum(p.numel() for p in self.seg_net.parameters())

        logger.info(f"seg_net has {num_params} parameters")

        self.model_dir = os.path.join(app_dir, "model")
        self.pretrained_model = os.path.join(self.model_dir, "pretrained.pt")
        self.final_model = os.path.join(self.model_dir, "model.pt")

        self.image_size = 256

        super().__init__(
            app_dir=app_dir,
            studies=studies,
            conf=conf,
            name="LungAIR Segmentation",
            description="Basic UNet Segmentation for LungAIR",
        )

    def init_infers(self) -> Dict[str, InferTask]:
        infers = {
            "segmentation": MyInfer([self.pretrained_model, self.final_model], self.seg_net, self.image_size),
            # intensity range set for CT Soft Tissue; TODO: see if you can find suitable settings for lungs in a CXR.
            "Histogram+GraphCut": HistogramBasedGraphCut(
                intensity_range=(-300, 200, 0.0, 1.0, True), pix_dim=(2.5, 2.5, 5.0), lamda=1.0, sigma=0.1
            ),
        }

        # Simple way to Add deepgrow 2D+3D models for infer tasks
        # infers.update(self.deepgrow_infer_tasks(self.model_dir)) # TODO: decide whether to keep this
        return infers

    def init_trainers(self) -> Dict[str, TrainTask]:
        return {
            "segmentation": MyTrain(
                model_dir=self.model_dir,
                network=self.seg_net,
                image_size = self.image_size,
                load_path=self.pretrained_model, # TODO: I wonder what happens when there isn't anything at this path?
                publish_path=self.final_model,
                config={"max_epochs": 100, "train_batch_size": 4, "to_gpu": True},
            ),
        }

    def init_strategies(self) -> Dict[str, Strategy]:
        strategies: Dict[str, Strategy] = {}
        strategies["random"] = Random()
        # strategies["CustomStrategy"] = MyStrategy() # TODO consider making another strategy in lib/strategy.py and uncommenting this.
        return strategies

    def init_scoring_methods(self) -> Dict[str, ScoringMethod]:
        methods: Dict[str, ScoringMethod] = {}
        return methods # Does nothing right now.
