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
from typing import Callable, Sequence

import monai

from monailabel.interfaces.tasks.infer import InferTask, InferType
from monailabel.transform.post import BoundingBoxd, Restored


class MyInfer(InferTask):
    """
    Custom infer task
    """

    def __init__(
        self,
        path,
        network,
        image_size,
        # I believe the following are just for logging a description
        labels=["background", "lung"],
        dimension=2,
        description="LungAIR infer task",
        type=InferType.SEGMENTATION, # This one only seems to matter if it's SCRIBBLES
    ):
        super().__init__(
            path=path,
            network=network,
            type=type,
            labels=labels,
            dimension=dimension,
            description=description,
        )
        self.image_size = image_size

    def pre_transforms(self, data=None) -> Sequence[Callable]:
        keys = "image"
        return [
            monai.transforms.LoadImageD(reader='itkreader', keys=keys),
            monai.transforms.TransposeD(indices = (2,1,0), keys=keys),
            monai.transforms.ResizeD(
                spatial_size=(self.image_size,self.image_size),
                mode = "bilinear",
                align_corners = False,
                keys=keys
            ),
            monai.transforms.ToTensorD(keys=keys),
            monai.transforms.AddChanneld(keys=keys),
        ]

    def inferer(self, data=None) -> Callable:
        return monai.inferers.SimpleInferer()

    def post_transforms(self, data=None) -> Sequence[Callable]:
        return [
            monai.transforms.ActivationsD(keys="pred", softmax=True),
            monai.transforms.AsDiscreteD(keys="pred", argmax=True),
            monai.transforms.ToNumpyd(keys="pred"),
        ]
