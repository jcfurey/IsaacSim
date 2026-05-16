#!/bin/bash

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
