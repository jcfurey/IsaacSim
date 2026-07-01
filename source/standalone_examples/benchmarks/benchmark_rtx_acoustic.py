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

"""Benchmark RTX acoustic sensor performance in Isaac Sim.

Measures frame time and memory (including the "GPU Memory Dedicated" metric
recorded by the benchmark framework) for a configurable number of RTX
acoustic sensors. To estimate per-sensor VRAM cost, run this benchmark with
increasing ``--num-sensors`` values (e.g. 1, 2, 4) and compare the recorded
GPU memory between runs; the slope is the per-sensor cost, and the
``--num-sensors 0``-less baseline can be approximated by the loading phase
measurements.

Unlike lidar and radar, acoustic sensors produce no point cloud, so no
debug-draw writer exists for them. Instead, the GenericModelOutput annotator
is attached through the AcousticSensor runtime class, which drives the full
acoustic data pipeline (ray tracing + waveform synthesis + GMO readback)
every frame.
"""

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--num-sensors", type=int, default=1, help="Number of sensors")
parser.add_argument("--num-gpus", type=int, default=None, help="Number of GPUs on machine.")
parser.add_argument("--num-frames", type=int, default=600, help="Number of frames to run benchmark for")
parser.add_argument(
    "--backend-type",
    default="OmniPerfKPIFile",
    choices=["LocalLogMetrics", "JSONFileMetrics", "OsmoKPIFile", "OmniPerfKPIFile"],
    help="Benchmarking backend, defaults",
)

args, unknown = parser.parse_known_args()

n_sensor = args.num_sensors
n_gpu = args.num_gpus
n_frames = args.num_frames

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True, "max_gpu_count": n_gpu})

import carb
import omni
from isaacsim.core.utils.extensions import enable_extension
from isaacsim.sensors.experimental.rtx import Acoustic, AcousticSensor, parse_generic_model_output_data

enable_extension("isaacsim.benchmark.services")
from isaacsim.benchmark.services import BaseIsaacBenchmark

# Create the benchmark
benchmark = BaseIsaacBenchmark(
    benchmark_name="benchmark_rtx_acoustic",
    workflow_metadata={
        "metadata": [
            {"name": "num_acoustic_sensors", "data": n_sensor},
            {"name": "num_gpus", "data": carb.settings.get_settings().get("/renderer/multiGpu/currentGpuCount")},
        ]
    },
    backend_type=args.backend_type,
)
benchmark.set_phase("loading", start_recording_frametime=False, start_recording_runtime=True)

scene_path = "/Isaac/Environments/Simple_Warehouse/full_warehouse.usd"
benchmark.fully_load_stage(benchmark.assets_root_path + scene_path)
timeline = omni.timeline.get_timeline_interface()

sensors = []
for i in range(n_sensor):
    acoustic_path = "/World/RtxAcoustic_" + str(i)
    # these positions are used for full_warehouse.usd (same rack aisle as the
    # lidar benchmark, so there is nearby geometry to reflect off)
    sensor_translation = [-8.0, 13.0 + i * 2.0, 2.0]

    # Multi-transmitter/receiver topology (three mounts, two receiver groups)
    # matching the extension's own runtime tests, so the benchmark exercises a
    # realistic multi-signal-way workload rather than the minimal default.
    acoustic = Acoustic(
        acoustic_path,
        tick_rate=30.0,
        translations=sensor_translation,
        orientations=[1.0, 0.0, 0.0, 0.0],  # wxyz
        attributes={
            "omni:sensor:WpmAcoustic:centerFrequency": 51200.0,
            "omni:sensor:WpmAcoustic:sensorMount:m001:position": (0.0, -0.05, 0.0),
            "omni:sensor:WpmAcoustic:sensorMount:m001:rotation": (0.0, 0.0, 0.0),
            "omni:sensor:WpmAcoustic:sensorMount:m002:position": (0.0, 0.0, 0.0),
            "omni:sensor:WpmAcoustic:sensorMount:m002:rotation": (0.0, 0.0, 0.0),
            "omni:sensor:WpmAcoustic:sensorMount:m003:position": (0.0, 0.05, 0.0),
            "omni:sensor:WpmAcoustic:sensorMount:m003:rotation": (0.0, 0.0, 0.0),
            "omni:sensor:WpmAcoustic:rxGroup:g001:receiverIndices": [0, 1],
            "omni:sensor:WpmAcoustic:rxGroup:g002:receiverIndices": [1, 2],
        },
    )
    sensor = AcousticSensor(acoustic, annotators=["generic-model-output"])
    sensors.append(sensor)

    omni.kit.app.get_app().update()

benchmark.store_measurements()

benchmark.set_phase("benchmark", warmup_frames=15)
timeline.play()

for _ in range(1, n_frames):
    omni.kit.app.get_app().update()

benchmark.store_measurements()
benchmark.stop()

# Sanity-check that the acoustic pipeline actually produced data during the
# run: an all-zero output would mean the benchmark timed an idle pipeline.
for i, sensor in enumerate(sensors):
    data, _ = sensor.get_data("generic-model-output")
    num_elements = parse_generic_model_output_data(data).numElements if data is not None else 0
    print(f"[benchmark_rtx_acoustic] sensor {i}: numElements={num_elements}")

timeline.stop()

for sensor in sensors:
    sensor.destroy()
omni.kit.app.get_app().update()

simulation_app.close()
