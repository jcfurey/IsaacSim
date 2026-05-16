# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Kit command helpers for importing URDF from ROS 2 nodes."""

import os
import typing
from functools import partial

import carb
import omni.client
import omni.kit.commands
from isaacsim.asset.importer.urdf import URDFImporter, URDFImporterConfig
from isaacsim.ros2.urdf.robot_definition_reader import RobotDefinitionReader
from omni.client import Result
from pxr import Usd, UsdUtils


class URDFImportFromROS2Node(omni.kit.commands.Command):
    """Command that imports a URDF from a ROS 2 node.

    Args:
        ros2_node_name: ROS 2 node to query for robot_description.
        import_config: Import configuration overrides.
        dest_path: Destination path for output assets.
        get_articulation_root: Whether to return articulation root.
    """

    def __init__(
        self,
        ros2_node_name: str = "robot_state_publisher",
        import_config: typing.Optional[URDFImporterConfig] = None,
        dest_path: str = "",
        get_articulation_root: bool = False,
    ):
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
        self.__subscription = carb.eventdispatcher.get_eventdispatcher().observe_event(
            event_name=omni.kit.app.GLOBAL_EVENT_UPDATE,
            on_event=self.on_app_update,
            observer_name="isaacsim.ros2.urdf.commands.URDFImportFromROS2Node._on_app_update",
        )

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

    def on_description_received(self, urdf_description: str) -> None:
        """Persist the received URDF description to disk.

        Args:
            urdf_description: URDF document string from the node.
        """
        ext_manager = omni.kit.app.get_app().get_extension_manager()
        ext_id = ext_manager.get_enabled_extension_id("isaacsim.ros2.urdf")
        self._extension_path = ext_manager.get_extension_path(ext_id)
        data_folder = os.path.join(self._extension_path, "data", "urdf", "temp")
        os.makedirs(data_folder, exist_ok=True)
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
