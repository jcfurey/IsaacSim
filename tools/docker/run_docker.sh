#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

PRIVACY_EMAIL="${PRIVACY_EMAIL:-user@example.com}"  # Allow override via environment

# ROS 2 bridge env passthrough. Only forwarded if the user set the var on the
# host (-e VAR with no value picks up the host value); unset vars are ignored
# by docker, so the container falls back to setup_ros_env.sh defaults.
docker run --name isaac-sim --rm -it --gpus all --network=host \
  --entrypoint bash \
  -e ACCEPT_EULA=Y \
  -e OMNI_ENV_PRIVACY_CONSENT=Y \
  -e OMNI_ENV_PRIVACY_USERID="${PRIVACY_EMAIL}" \
  -e ROS_DISTRO \
  -e RMW_IMPLEMENTATION \
  -e ROS_DOMAIN_ID \
  -e FASTRTPS_DEFAULT_PROFILES_FILE \
  isaac-sim-docker:latest \
  "$@"
