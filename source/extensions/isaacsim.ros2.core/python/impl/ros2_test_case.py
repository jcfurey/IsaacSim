# SPDX-FileCopyrightText: Copyright (c) 2018-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import asyncio
import threading

import carb
import omni
from isaacsim.core.experimental.utils import stage as stage_utils
from isaacsim.core.simulation_manager import SimulationManager

try:
    from isaacsim.test.utils import TimedAsyncTestCase
except ImportError:
    raise ImportError(
        "isaacsim.test.utils is required to use ROS2TestCase. " "This module should only be imported in test contexts."
    )


class ROS2TestCase(TimedAsyncTestCase):
    """Base test class that automatically times all test methods.

    This class extends TimedAsyncTestCase to add ROS2 specific setup and teardown.
    It also provides helper methods for creating and managing ROS2 resources with automatic cleanup.
    """

    async def setUp(self):
        """Set up test timing before each test method."""
        await super().setUp()
        self._timeline = omni.timeline.get_timeline_interface()

        # Initialize lists to track ROS2 resources for automatic cleanup
        self._ros2_nodes = []
        self._ros2_publishers = []
        self._ros2_subscribers = []
        self._ros2_executors = {}  # node -> (executor, thread)
        self._ros2_callback_groups = {}  # node -> ReentrantCallbackGroup

        import rclpy

        if not rclpy.ok():
            rclpy.init()

        SimulationManager.set_backend("numpy")
        SimulationManager.set_physics_sim_device("cpu")
        SimulationManager.enable_fabric(enable=False)
        await omni.kit.app.get_app().next_update_async()
        await stage_utils.create_new_stage_async()
        try:
            from isaacsim.core.rendering_manager import ViewportManager

            status, frames = await ViewportManager.wait_for_viewport_async()
            self.assertTrue(status, f"Viewport not ready after {frames} frames")
        except Exception:
            pass

    def create_node(self, node_name):
        """Create a ROS2 node and track it for automatic cleanup.

        Args:
            node_name: Name of the ROS2 node to create.

        Returns:
            The created ROS2 node.
        """
        import rclpy

        node = rclpy.create_node(node_name)
        self._ros2_nodes.append(node)
        return node

    def create_publisher(self, node, msg_type, topic_name, qos_profile=10):
        """Create a ROS2 publisher and track it for automatic cleanup.

        Args:
            node: The ROS2 node to create the publisher on.
            msg_type: The message type for the publisher.
            topic_name: The topic name to publish to.
            qos_profile: QoS profile for the publisher (default: 10).

        Returns:
            The created ROS2 publisher.
        """
        publisher = node.create_publisher(msg_type, topic_name, qos_profile)
        self._ros2_publishers.append((node, publisher))
        return publisher

    def create_subscription(self, node, msg_type, topic_name, callback, qos_profile=10):
        """Create a ROS2 subscription and track it for automatic cleanup.

        When the node has a background executor (via start_async_spinning), a
        ReentrantCallbackGroup is used automatically so multiple subscriptions
        can fire in parallel.

        Args:
            node: The ROS2 node to create the subscription on.
            msg_type: The message type for the subscription.
            topic_name: The topic name to subscribe to.
            callback: Callback function for received messages.
            qos_profile: QoS profile for the subscription (default: 10).

        Returns:
            The created ROS2 subscription.
        """
        cb_group = self._ros2_callback_groups.get(node)
        subscription = node.create_subscription(msg_type, topic_name, callback, qos_profile, callback_group=cb_group)
        self._ros2_subscribers.append((node, subscription))
        return subscription

    def destroy_subscription(self, node, subscription):
        """Manually destroy a subscription and remove it from tracking.

        Args:
            node: The ROS2 node that owns the subscription.
            subscription: The subscription to destroy.
        """
        try:
            node.destroy_subscription(subscription)
            # Remove from tracking list to avoid double cleanup
            if (node, subscription) in self._ros2_subscribers:
                self._ros2_subscribers.remove((node, subscription))
        except Exception as e:
            carb.log_warn(f"Failed to destroy subscription: {e}")

    def destroy_publisher(self, node, publisher):
        """Manually destroy a publisher and remove it from tracking.

        Args:
            node: The ROS2 node that owns the publisher.
            publisher: The publisher to destroy.
        """
        try:
            node.destroy_publisher(publisher)
            # Remove from tracking list to avoid double cleanup
            if (node, publisher) in self._ros2_publishers:
                self._ros2_publishers.remove((node, publisher))
        except Exception as e:
            carb.log_warn(f"Failed to destroy publisher: {e}")

    def start_async_spinning(self, node):
        """Start a background executor that continuously spins the node.

        Callbacks on this node will fire automatically in a background thread,
        removing the need for manual spin_once() calls in frame loops.
        A ReentrantCallbackGroup is created so subscriptions added via
        create_subscription can fire in parallel.

        Args:
            node: The ROS2 node to spin.
        """
        from rclpy.callback_groups import ReentrantCallbackGroup
        from rclpy.executors import MultiThreadedExecutor

        if node in self._ros2_executors:
            carb.log_warn(f"node {node.get_name()} is already spinning, skipping")
            return

        self._ros2_callback_groups[node] = ReentrantCallbackGroup()

        executor = MultiThreadedExecutor()
        executor.add_node(node)
        thread = threading.Thread(target=executor.spin, daemon=True)
        thread.start()
        self._ros2_executors[node] = (executor, thread)

    def stop_async_spinning(self, node):
        """Stop the background executor for a node.

        Args:
            node: The ROS2 node to stop spinning.
        """
        if node not in self._ros2_executors:
            return
        executor, thread = self._ros2_executors.pop(node)
        self._ros2_callback_groups.pop(node, None)
        executor.shutdown()
        thread.join(timeout=5.0)

    async def simulate_until_condition(
        self, condition_func, max_frames=180, frames_per_step=1, per_frame_callback=None
    ):
        """Simulate until condition is met or maximum frames reached.

        This method runs simulation in steps until a specified condition function
        returns True or the maximum frame limit is exceeded.

        Args:
            condition_func: Function that returns True when condition is met.
            max_frames: Maximum number of simulation frames to run.
            frames_per_step: Number of frames to simulate in each step.
            per_frame_callback: Optional callback to execute each frame (e.g., for spinning ROS nodes).

        Returns:
            True if condition was met, False if max frames reached.
        """
        frames_run = 0
        while frames_run < max_frames:
            await omni.kit.app.get_app().next_update_async()
            if per_frame_callback is not None:
                per_frame_callback()
            frames_run += frames_per_step
            if condition_func():
                return True
        return False

    async def tearDown(self):
        self._timeline.stop()
        await omni.kit.app.get_app().next_update_async()
        while omni.usd.get_context().get_stage_loading_status()[2] > 0:
            carb.log_info("tearDown, assets still loading, waiting to finish...")
            await asyncio.sleep(1.0)

        # Clean up ROS2 resources in the correct order
        # First stop any background executors
        for node, (executor, thread) in list(self._ros2_executors.items()):
            try:
                executor.shutdown()
                thread.join(timeout=5.0)
            except Exception as e:
                carb.log_warn(f"Failed to stop executor: {e}")
        self._ros2_executors.clear()

        # Then destroy publishers
        for node, publisher in self._ros2_publishers:
            try:
                node.destroy_publisher(publisher)
            except Exception as e:
                carb.log_warn(f"Failed to destroy publisher: {e}")

        # Then destroy subscribers
        for node, subscription in self._ros2_subscribers:
            try:
                node.destroy_subscription(subscription)
            except Exception as e:
                carb.log_warn(f"Failed to destroy subscription: {e}")

        # Finally destroy nodes
        for node in self._ros2_nodes:
            try:
                node.destroy_node()
            except Exception as e:
                carb.log_warn(f"Failed to destroy node: {e}")

        # Clear the tracking lists
        self._ros2_publishers.clear()
        self._ros2_subscribers.clear()
        self._ros2_nodes.clear()

        import rclpy

        if rclpy.ok():
            rclpy.shutdown()

        await super().tearDown()
