#!/bin/bash -e

# Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Download checkpoint

mkdir -p ${CHECKPOINT_DIR}
if [ -f "${CHECKPOINT_DIR}/checkpoint" ]; then 
  echo "Checkpoint already downloaded."
else
  echo "Downloading checkpoint ..."
  wget --content-disposition https://api.ngc.nvidia.com/v2/models/nvidia/rn50_tf_amp_ckpt/versions/20.06.0/zip -O \
    rn50_tf_amp_ckpt_20.06.0.zip || {
    echo "ERROR: Failed to download checkpoint from NGC"
    exit 1
  }
  unzip rn50_tf_amp_ckpt_20.06.0.zip -d ${CHECKPOINT_DIR}
  rm rn50_tf_amp_ckpt_20.06.0.zip
  echo "ok"
fi
