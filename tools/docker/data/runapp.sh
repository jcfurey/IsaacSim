#! /bin/sh

# Set up ROS 2 environment (LD_LIBRARY_PATH for bundled distro libs, RMW_IMPLEMENTATION)
# so the isaacsim.ros2.bridge extension can be enabled without manual host setup.
# No-op unless the user enables the bridge; safe to source unconditionally.
if [ -f /isaac-sim/setup_ros_env.sh ]; then
    . /isaac-sim/setup_ros_env.sh
fi

# Optional flags from env (only add if set)
EXTRA_FLAGS=""
[ -n "${OMNI_SERVER}" ] && EXTRA_FLAGS="${EXTRA_FLAGS} --/persistent/isaac/asset_root/default=${OMNI_SERVER}"

/isaac-sim/license.sh && /isaac-sim/privacy.sh && /isaac-sim/isaac-sim.sh \
  --merge-config="/isaac-sim/config/open_endpoint.toml" \
  $EXTRA_FLAGS \
  "$@"
