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

import monai
import numpy as np

from monailabel.tasks.train.basic_train import BasicTrainTask, Context

logger = logging.getLogger(__name__)


class MyTrain(BasicTrainTask):
    def __init__(
        self,
        model_dir,
        network,
        image_size,
        description="Train generic Segmentation model",
        **kwargs,
    ):
        self._network = network
        self.image_size = image_size
        super().__init__(model_dir, description, **kwargs)

    def network(self, context: Context):
        return self._network

    def optimizer(self, context: Context):
        return monai.optimizers.Novograd(self._network.parameters(), 0.0001)

    def loss_function(self, context: Context):
        return monai.losses.DiceLoss(
            # unlike in the jupyter notebook version of our lung seg exploration, here the labels are expected to not be in one-hot form
            # (TODO: pay attention to this part and make sure it's correct)
            to_onehot_y = True,
            # Note that our segmentation network is missing the softmax at the end. However a softmax occurs below
            # in the train_post_transforms. However train_post_transforms doesn't seem to matter for this.
            # Maybe train_post_transforms is for transforms to apply in order to examine the result of training? Unclear.
            softmax = True,
        )

    def train_pre_transforms(self, context: Context):
        keys = ['image', 'label']
        sampling_modes = ['blinear', 'nearest']
        align_corners = [False, None]
        t = [
            monai.transforms.LoadImageD(reader='itkreader', keys=keys),
            monai.transforms.TransposeD(indices = (2,1,0), keys=keys),
            monai.transforms.ResizeD(
                spatial_size=(self.image_size,self.image_size),
                mode = sampling_modes,
                align_corners = align_corners,
                keys=keys
            ),
            monai.transforms.ToTensorD(keys=keys),
        ]
        if context.request.get("to_gpu", False):
            t.extend([monai.transforms.ToDeviceD(keys=keys, device=context.device)])
        t.extend(
            [
                monai.transforms.RandZoomD( keys=keys,
                    mode = sampling_modes,
                    align_corners = align_corners,
                    prob=1.,
                    padding_mode="constant",
                    min_zoom = 0.7,
                    max_zoom=1.3,
                ),
                monai.transforms.RandRotateD( keys=keys,
                    mode = sampling_modes,
                    align_corners = align_corners,
                    prob=1.,
                    range_x = np.pi/8,
                    padding_mode="zeros",
                ),
                monai.transforms.RandGaussianSmoothD( keys=keys,
                    prob = 0.4
                ),
                monai.transforms.RandAdjustContrastD( keys=keys,
                    prob=0.4,
                ),
            ]
        )
        return t

    def train_post_transforms(self, context: Context):
        return [
            monai.transforms.ToTensorD(keys=("pred", "label")),
            monai.transforms.ActivationsD(keys="pred", softmax=True),
            monai.transforms.AsDiscreteD(
                keys=("pred", "label"),
                argmax=(True, False),
                to_onehot=2,
            ),
        ]

    def val_pre_transforms(self, context: Context):
        keys = ['image', 'label']
        sampling_modes = ['blinear', 'nearest']
        align_corners = [False, None]
        return [
            monai.transforms.LoadImageD(reader='itkreader', keys=keys),
            monai.transforms.TransposeD(indices = (2,1,0), keys=keys),
            monai.transforms.ResizeD(
                spatial_size=(self.image_size,self.image_size),
                mode = sampling_modes,
                align_corners = align_corners,
                keys=keys
            ),
            monai.transforms.ToTensorD(keys=keys),
            monai.transforms.AddChanneld(keys=("image")),
            monai.transforms.ToDeviceD(keys=("image", "label"), device=context.device),
        ]

    # Not overriding val_post_transforms means we accept that val_post_transforms simply
    # calls train_post_transforms; i.e. it does the argmax etc.

    def val_inferer(self, context: Context):
        return monai.inferers.SimpleInferer()
