# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Deprecated Kit command helpers for importing URDF from ROS 2 nodes."""

import os
import tempfile
import time
import typing
from functools import partial

import carb
import omni.client
import omni.kit.commands
from isaacsim.asset.importer.urdf import URDFImporter, URDFImporterConfig
from isaacsim.ros2.urdf.robot_definition_reader import RobotDefinitionReader
from omni.client import Result


class URDFImportFromROS2Node(omni.kit.commands.Command):
    """Deprecated command that imports a URDF from a ROS 2 node.

    .. deprecated:: Use RobotDefinitionReader and URDFImporter directly instead.

    Args:
        ros2_node_name: ROS 2 node to query for robot_description.
        import_config: Import configuration overrides.
        dest_path: Destination path for output assets.
        get_articulation_root: Whether to return articulation root.
    """

    # How long to keep polling for the robot_description before releasing the
    # app-update observer (which is what keeps this command object alive).
    _RESPONSE_TIMEOUT_SEC = 60.0

    def __init__(
        self,
        ros2_node_name: str = "robot_state_publisher",
        import_config: typing.Optional[URDFImporterConfig] = None,
        dest_path: str = "",
        get_articulation_root: bool = False,
    ) -> None:
        carb.log_warn(
            "URDFImportFromROS2Node command is deprecated and will be removed in a future version. "
            "Use RobotDefinitionReader and URDFImporter classes directly instead."
        )
        self.urdf_importer = URDFImporter()
        self.ros2_node_name = ros2_node_name
        self.dest_path = dest_path
        # Avoid the mutable-default-argument trap: a URDFImporterConfig() in the
        # parameter list is evaluated once at class definition time and would be
        # shared across every call that omits import_config. Subsequent commands
        # would then see urdf_path / usd_path mutations from prior runs leak in.
        self.config = import_config if import_config is not None else URDFImporterConfig()
        self.robot_definition = RobotDefinitionReader()
        self.robot_definition.description_received_fn = partial(self.on_description_received)
        self.urdf_path = None
        self.finished = False
        # The observer holds a strong reference to the bound method (and thus
        # to this command object) -- that is the intended keep-alive while the
        # asynchronous robot_description fetch is in flight. It also means
        # __del__ can never free anything while the subscription exists, so
        # the never-responds case must be handled by the timeout below, which
        # releases the subscription (and with it the object) from inside the
        # callback itself.
        self._deadline = time.monotonic() + self._RESPONSE_TIMEOUT_SEC
        self.__subscription = carb.eventdispatcher.get_eventdispatcher().observe_event(
            event_name=omni.kit.app.GLOBAL_EVENT_UPDATE,
            on_event=self.on_app_update,
            observer_name="isaacsim.ros2.urdf.commands.URDFImportFromROS2Node._on_app_update",
        )

    def __del__(self) -> None:
        """Safety net for explicit deletion before completion or timeout."""
        self.__subscription = None

    def on_app_update(self, event: typing.Any) -> None:
        """Handle app update ticks to trigger import completion.

        Args:
            event: App update event payload.
        """
        if self.finished:
            self.__subscription = None
            if self.urdf_path:
                self.import_robot(self.urdf_path)
            return
        if time.monotonic() > self._deadline:
            # The ROS 2 node never answered: release the observer so this
            # command object stops ticking every frame and becomes
            # collectable. Without this, the dispatcher's strong reference to
            # the bound callback keeps the object (and the per-frame tick)
            # alive for the life of the process.
            carb.log_warn(
                f"URDFImportFromROS2Node: no robot_description received from "
                f"'{self.ros2_node_name}' within {self._RESPONSE_TIMEOUT_SEC} s; giving up."
            )
            self.__subscription = None
            return

    def on_description_received(self, urdf_description: str, package_found: bool = False) -> None:
        """Persist the received URDF description to disk.

        Args:
            urdf_description: URDF document string from the node.
            package_found: Whether ROS package URLs were resolved.
        """
        data_folder = tempfile.mkdtemp(prefix="ros2_urdf_cmd_")
        urdf_path = os.path.join(data_folder, "urdf_description.urdf")
        with open(urdf_path, "w", encoding="utf-8") as f:
            f.write(urdf_description)

        self.finished = True
        self.urdf_path = urdf_path

    def import_robot(self, urdf_path: str) -> None:
        """Import the robot from a URDF file.

        Args:
            urdf_path: Path to the URDF file to import.
        """
        self.config.urdf_path = urdf_path
        if self.dest_path:
            self.config.usd_path = self.dest_path
        self.urdf_importer.config = self.config
        self.urdf_importer.import_urdf()

    def do(self) -> Result:
        """Execute the command to fetch and import the URDF.

        Returns:
            Command result status.
        """
        self.robot_definition.start_get_robot_description(self.ros2_node_name)
        return Result.OK
