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

"""Dense sonar-style acoustic array on a SINGLE OmniAcoustic prim.

This example builds a dense multi-element ultrasonic array (default: 128
elements) to approximate an imaging-sonar sensor. The key design decision is
that all elements live on ONE OmniAcoustic prim as ``sensorMount``
multi-apply schema instances -- NOT as separate AcousticSensor objects:

- One prim means ONE render product / Hydra pipeline, instead of one per
  element. Per-render-product overhead (pipeline resources, GMO streams,
  scheduling) is the dominant fixed VRAM/perf cost of an RTX sensor, so 128
  separate sensors would multiply that cost 128x for no modeling benefit.
- The WPM acoustic model natively simulates multi-transmitter /
  multi-receiver topologies inside one trace pass: ray cost scales with the
  number of *firing* transmitters (rays are emitted per firing event), while
  additional receivers are nearly free. The GMO output labels every
  amplitude sample with (x=tx mount ID, y=rx mount ID, z=channel ID), giving
  the full signal-way matrix needed for beamforming.
- The firing sequence (``firingSeq`` schema instances) defines which
  transmitters fire, at what time offsets, on which channels, and which
  receiver group listens -- modeling a real sonar's scan cycle within one
  sensor frame.

Scaling knobs (per-frame cost):
- rays  ~ num_transmitters x (azSpanDeg*raysPerDeg) x (elSpanDeg*raysPerDeg)
- GMO   ~ num_signal_ways x samples_per_signal_way, where
  num_signal_ways ~ num_transmitters x num_receivers and
  samples_per_signal_way ~ frame_period / sampleDuration

If your target element count exceeds what the acoustic plugin supports on a
single prim, shard the array across a few prims (e.g. 4 prims x 32 mounts)
rather than one prim per element -- each prim still costs a render product.
"""

import argparse
import math
import os

from isaacsim import SimulationApp

parser = argparse.ArgumentParser(description="Dense sonar-style RTX Acoustic array example.")
parser.add_argument("--test", default=False, action="store_true", help="Run in test mode.")
parser.add_argument("--num-elements", type=int, default=128, help="Number of transducer elements in the array.")
parser.add_argument(
    "--num-transmitters",
    type=int,
    default=1,
    help="Number of elements that fire (spread evenly across the array); all elements receive.",
)
parser.add_argument(
    "--geometry",
    type=str,
    default="grid",
    choices=["line", "grid", "ring"],
    help="Array element layout.",
)
parser.add_argument(
    "--center-frequency", type=float, default=51200.0, help="Center frequency in Hz (sets lambda/2 element spacing)."
)
parser.add_argument(
    "--fire-interval-ms",
    type=float,
    default=2.0,
    help="Time offset between consecutive transmitter firings within a frame (ms).",
)
args, _ = parser.parse_known_args()

SPEED_OF_SOUND = 343.0  # m/s (air; the WPM acoustic model is an in-air ultrasonic model)


def generate_sonar_array_attributes(
    num_elements: int,
    num_transmitters: int,
    geometry: str,
    center_frequency: float,
    fire_interval_ms: float,
) -> dict:
    """Generate the OmniAcoustic attribute dict for a dense transducer array.

    Elements are laid out at half-wavelength spacing (the classic dense-array
    spacing that avoids grating lobes) in the sensor's local Y-Z plane, all
    facing +X. Transmitter elements are spread evenly across the array and
    fire sequentially (time-division) on distinct channels; a single receiver
    group contains every element.

    Args:
        num_elements: Total number of transducer elements (sensor mounts).
        num_transmitters: Number of elements that fire. Must be <= num_elements.
        geometry: ``"line"`` (along Y), ``"grid"`` (Y-Z plane, near-square),
            or ``"ring"`` (circle in the Y-Z plane).
        center_frequency: Center frequency in Hz; element spacing is lambda/2.
        fire_interval_ms: Offset between consecutive firing events in ms.

    Returns:
        Attribute mapping for the ``Acoustic`` authoring class, containing
        per-element ``sensorMount`` instances, one ``rxGroup`` with all
        elements, and one ``firingSeq`` with the transmitter schedule.
    """
    if not 1 <= num_transmitters <= num_elements:
        raise ValueError(f"num_transmitters ({num_transmitters}) must be in [1, num_elements={num_elements}]")
    spacing = SPEED_OF_SOUND / center_frequency / 2.0  # lambda/2

    # element positions in the sensor's local frame (Y-Z plane, facing +X)
    positions = []
    if geometry == "line":
        for i in range(num_elements):
            y = (i - (num_elements - 1) / 2.0) * spacing
            positions.append((0.0, y, 0.0))
    elif geometry == "grid":
        cols = int(math.ceil(math.sqrt(num_elements)))
        rows = int(math.ceil(num_elements / cols))
        for i in range(num_elements):
            r, c = divmod(i, cols)
            y = (c - (cols - 1) / 2.0) * spacing
            z = (r - (rows - 1) / 2.0) * spacing
            positions.append((0.0, y, z))
    elif geometry == "ring":
        # circumference = num_elements * spacing -> radius for lambda/2 arc spacing
        radius = num_elements * spacing / (2.0 * math.pi)
        for i in range(num_elements):
            angle = 2.0 * math.pi * i / num_elements
            positions.append((0.0, radius * math.cos(angle), radius * math.sin(angle)))
    else:
        raise ValueError(f"Unknown geometry '{geometry}'")

    attributes = {"omni:sensor:WpmAcoustic:centerFrequency": center_frequency}
    for i, position in enumerate(positions):
        mount = f"m{i + 1:03d}"
        attributes[f"omni:sensor:WpmAcoustic:sensorMount:{mount}:position"] = position
        attributes[f"omni:sensor:WpmAcoustic:sensorMount:{mount}:rotation"] = (0.0, 0.0, 0.0)

    # one receiver group containing every element (receiverIndices are 0-based
    # mount indices)
    attributes["omni:sensor:WpmAcoustic:rxGroup:g001:receiverIndices"] = list(range(num_elements))

    # firing sequence: num_transmitters elements spread evenly across the
    # array, firing sequentially (time-division) on distinct channels, all
    # received by receiver group 0
    if num_transmitters == 1:
        tx_indices = [0]
    else:
        tx_indices = [round(k * (num_elements - 1) / (num_transmitters - 1)) for k in range(num_transmitters)]
    attributes["omni:sensor:WpmAcoustic:firingSeq:seq001:txSensorId"] = tx_indices
    attributes["omni:sensor:WpmAcoustic:firingSeq:seq001:eventTimeNs"] = [
        k * fire_interval_ms * 1.0e6 for k in range(num_transmitters)
    ]
    attributes["omni:sensor:WpmAcoustic:firingSeq:seq001:channel"] = list(range(num_transmitters))
    attributes["omni:sensor:WpmAcoustic:firingSeq:seq001:rxGroupId"] = [0] * num_transmitters
    return attributes


simulation_app = SimulationApp({"headless": True})

output_dir = os.path.join(os.getcwd(), "_example_output_isaacsim.sensors.experimental.rtx", "create_acoustic_sonar_array")
os.makedirs(output_dir, exist_ok=True)

import numpy as np
import omni
import omni.replicator.core as rep
from isaacsim.core.experimental.objects import Cube
from isaacsim.sensors.experimental.rtx import Acoustic, AcousticSensor, parse_generic_model_output_data
from omni.replicator.core import Writer

# =============================================================================
# CREATE A SIMPLE SCENE
# =============================================================================

for i, (x, y) in enumerate([(3, 0), (5, 2), (4, -2)]):
    Cube(f"/World/target_{i}", positions=np.array([float(x), float(y), 0.5]), scales=np.array([1.0, 1.0, 2.0]))

print("Created 3 target cubes")

# =============================================================================
# CREATE THE DENSE ARRAY ON A SINGLE ACOUSTIC PRIM
# =============================================================================

attributes = generate_sonar_array_attributes(
    num_elements=args.num_elements,
    num_transmitters=args.num_transmitters,
    geometry=args.geometry,
    center_frequency=args.center_frequency,
    fire_interval_ms=args.fire_interval_ms,
)

acoustic = Acoustic(
    "/World/sonar_array",
    aux_output_level="BASIC",  # numSgws / numSamplesPerSgw in auxiliary data
    tick_rate=20.0,
    translations=np.array([0.0, 0.0, 0.5]),
    attributes=attributes,
)

sensor = AcousticSensor(acoustic, annotators=[])

spacing_mm = SPEED_OF_SOUND / args.center_frequency / 2.0 * 1e3
print(
    f"Created sonar array at {acoustic.paths[0]}: {args.num_elements} elements "
    f"({args.geometry}, {spacing_mm:.2f} mm spacing), {args.num_transmitters} transmitter(s), "
    f"expected signal ways per firing: {args.num_transmitters} x {args.num_elements} "
    f"= {args.num_transmitters * args.num_elements}"
)


# =============================================================================
# WRITER: ASSEMBLE THE BEAMFORMING-READY WAVEFORM MATRIX
# =============================================================================


class GmoSonarArrayWriter(Writer):
    """Writer that reassembles GMO samples into a (signal_ways, samples) matrix."""

    def __init__(self) -> None:
        self.data_structure = "renderProduct"
        self.annotators = [rep.annotators.get("GenericModelOutput")]
        self._reported = False

    def write(self, data: dict[str, object]) -> None:
        """Build the per-signal-way waveform matrix and report echo ranges once."""
        if "renderProducts" not in data:
            return
        for _rp_name, rp_data in data["renderProducts"].items():
            gmo_raw = rp_data.get("GenericModelOutput")
            if isinstance(gmo_raw, dict):
                gmo_raw = gmo_raw.get("data")
            gmo = parse_generic_model_output_data(gmo_raw)
            n = gmo.numElements
            if n == 0 or self._reported:
                continue
            self._reported = True

            samples_per_sgw = getattr(gmo, "numSamplesPerSgw", 0)
            if not samples_per_sgw:
                # without auxiliary data, derive the count from the first
                # signal-way boundary (samples of one signal way are contiguous)
                tx_ids = np.ctypeslib.as_array(gmo.x, shape=(n,))
                rx_ids = np.ctypeslib.as_array(gmo.y, shape=(n,))
                boundary = np.flatnonzero((np.diff(tx_ids) != 0) | (np.diff(rx_ids) != 0))
                samples_per_sgw = int(boundary[0] + 1) if boundary.size else n
            num_sgws = n // samples_per_sgw

            amplitudes = np.ctypeslib.as_array(gmo.scalar, shape=(n,))
            # rows: signal ways (tx-rx pairs); columns: time samples.
            # This is the matrix a delay-and-sum (or any) beamformer consumes.
            waveforms = amplitudes[: num_sgws * samples_per_sgw].reshape(num_sgws, samples_per_sgw)

            tx_ids = np.ctypeslib.as_array(gmo.x, shape=(n,))[::samples_per_sgw][:num_sgws]
            rx_ids = np.ctypeslib.as_array(gmo.y, shape=(n,))[::samples_per_sgw][:num_sgws]

            print(f"Waveform matrix: {waveforms.shape} (signal ways x samples)")
            print(f"  transmitters seen: {np.unique(tx_ids).tolist()}")
            print(f"  receivers seen:    {len(np.unique(rx_ids))} unique elements")

            # crude single-echo range estimate per signal way, from the peak
            # sample index: range = t_peak * c / 2
            sample_duration = 0.0001024  # schema default omni:sensor:WpmAcoustic:sampleDuration
            peak_idx = np.argmax(np.abs(waveforms), axis=1)
            ranges = peak_idx * sample_duration * SPEED_OF_SOUND / 2.0
            print(f"  echo range estimate: median {np.median(ranges):.2f} m across {num_sgws} signal ways")


rep.WriterRegistry.register(GmoSonarArrayWriter)
sensor.attach_writer("GmoSonarArrayWriter")

# =============================================================================
# RUN SIMULATION
# =============================================================================
if args.test:
    stage = omni.usd.get_context().get_stage()
    stage.Export(os.path.join(output_dir, "stage.usda"))

timeline = omni.timeline.get_timeline_interface()
timeline.play()

frame_count = 0
while simulation_app.is_running() and (not args.test or frame_count < 20):
    simulation_app.update()
    frame_count += 1

timeline.stop()
sensor.destroy()
simulation_app.close()
