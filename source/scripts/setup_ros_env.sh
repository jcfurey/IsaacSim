#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Script to check ROS environment and source internal libraries if needed

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ISAAC_SIM_ROOT="$SCRIPT_DIR" 

DEFAULT_ROS_DISTRO="jazzy"

# Check Ubuntu version
if [ -f /etc/os-release ]; then
    . /etc/os-release
    UBUNTU_VERSION=$(echo $VERSION_ID)
    
    if [[ "$UBUNTU_VERSION" == "22.04" ]]; then
        DEFAULT_ROS_DISTRO="humble"
    elif [[ "$UBUNTU_VERSION" == "24.04" ]]; then
        DEFAULT_ROS_DISTRO="jazzy"
    fi
fi

# Default ROS_DISTRO from the Ubuntu version when the caller did not set one.
if [ -z "$ROS_DISTRO" ]; then
    export ROS_DISTRO="$DEFAULT_ROS_DISTRO"
fi

# Add the bundled distro's libraries to LD_LIBRARY_PATH for the effective ROS_DISTRO.
# This must run even when ROS_DISTRO was explicitly provided (e.g. forced via the
# environment to pick a specific bundled distro in the Docker image); previously this
# block only ran when ROS_DISTRO was unset, so forcing a distro silently left the
# bridge libraries off the path and the ROS 2 bridge failed to load. Skip it when a
# system ROS install is already on the path (the user sourced their own ROS), and
# avoid adding the same directory twice.
BRIDGE_EXT_PATH="$ISAAC_SIM_ROOT/exts/isaacsim.ros2.core"
BUNDLED_LIB_DIR="$BRIDGE_EXT_PATH/$ROS_DISTRO/lib"
if [ -d "$BUNDLED_LIB_DIR" ] && \
   [[ ":$LD_LIBRARY_PATH:" != *":$BUNDLED_LIB_DIR:"* ]] && \
   [[ ":$LD_LIBRARY_PATH:" != *"/opt/ros/"* ]]; then
    if [ -n "$LD_LIBRARY_PATH" ]; then
        export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$BUNDLED_LIB_DIR"
    else
        export LD_LIBRARY_PATH="$BUNDLED_LIB_DIR"
    fi
fi

# Set RMW implementation to FastDDS if not already set
if [ -z "$RMW_IMPLEMENTATION" ]; then
    export RMW_IMPLEMENTATION="rmw_fastrtps_cpp"
fi
