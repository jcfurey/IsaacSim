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

"""Multibeam-sonar-style RTX lidar: idealized per-beam ranges at sonar geometry.

This example configures a generic RTX lidar (a single OmniLidar prim with a
custom solid-state emitter pattern) as the GEOMETRY of a multibeam imaging
sonar, and collapses its returns into the per-beam range scanline such a
sonar outputs. It is the complement of ``create_acoustic_sonar_array.py``:

- The lidar path gives idealized beam-space output: exact per-beam ranges at
  the sensor's true angular resolution and full update rate, because rays
  are instantaneous and geometric. Use it when a consumer (nav stack,
  mapping, RL) needs clean multibeam range data.
- The acoustic path (see ``create_acoustic_sonar_array.py``) gives physical
  echo character -- envelope waveforms, multipath, occlusion lobes -- but
  with ~5 deg effective (envelope) beams and a time-of-flight-limited update
  rate. Use it when echo/waveform realism matters more than angular
  resolution.

The ``--preset oculus-m3000d-lf`` configuration reproduces the Blueprint
Subsea Oculus M3000d low-frequency mode geometry (130 deg x 20 deg
aperture, 512 beams at ~0.25 deg separation, 0.1-30 m range, 40 Hz update,
2.5 mm range resolution): 512 azimuth beams x 9 elevation rows spanning the
20 deg vertical fan, with each azimuth column collapsed to its nearest
return -- mimicking how a multibeam sonar reports the first echo within its
vertical fan per beam.

Fidelity notes:

- Rays are IDEALIZED (thin, instantaneous, geometric). There is no acoustic
  propagation: no multipath, no beam sidelobes, no speckle, no vertical-fan
  amplitude weighting, and update rate is not bound by two-way travel time
  (which is what allows the real unit's 40 Hz here, unlike the acoustic
  in-air model's ~5 Hz at 30 m).
- The 20 deg vertical fan is sampled at a finite number of elevation rows
  (9 by default, 2.5 deg apart); a small target sitting exactly between
  rows can be missed, whereas the real sonar's continuous fan would see it.
  Raise ``--num-elevation-rows`` for denser vertical coverage.
- Surface reflectance comes from the lidar material model (wavelength in
  nm), not acoustic impedance.
"""

from __future__ import annotations

import argparse
import math
import os

# Preset reproducing the Oculus M3000d low-frequency mode geometry.
PRESETS = {
    "oculus-m3000d-lf": dict(
        num_beams=512,
        az_span_deg=130.0,
        num_elevation_rows=9,
        el_span_deg=20.0,
        near_range=0.1,
        far_range=30.0,
        scan_rate_hz=40,
        range_resolution=0.0025,
    ),
}

# two-stage parsing so a preset provides defaults while explicit flags win
_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--preset", type=str, default=None, choices=sorted(PRESETS.keys()))
_pre_args, _ = _pre.parse_known_args()

parser = argparse.ArgumentParser(description="Multibeam-sonar-style RTX lidar example.", parents=[_pre])
parser.add_argument("--test", default=False, action="store_true", help="Run in test mode.")
parser.add_argument("--num-beams", type=int, default=256, help="Number of azimuth beams across the fan.")
parser.add_argument("--az-span-deg", type=float, default=90.0, help="Horizontal (azimuth) fan width in degrees.")
parser.add_argument("--num-elevation-rows", type=int, default=5, help="Elevation rows sampling the vertical fan.")
parser.add_argument("--el-span-deg", type=float, default=20.0, help="Vertical fan width in degrees.")
parser.add_argument("--near-range", type=float, default=0.1, help="Minimum range in meters.")
parser.add_argument("--far-range", type=float, default=30.0, help="Maximum range in meters.")
parser.add_argument("--scan-rate-hz", type=int, default=20, help="Scan (and tick) rate in Hz.")
parser.add_argument("--range-resolution", type=float, default=0.0025, help="Range resolution in meters.")
if _pre_args.preset is not None:
    parser.set_defaults(**PRESETS[_pre_args.preset])
args, _ = parser.parse_known_args()


def generate_multibeam_lidar_attributes(
    num_beams: int,
    az_span_deg: float,
    num_elevation_rows: int,
    el_span_deg: float,
    near_range: float,
    far_range: float,
    scan_rate_hz: int,
    range_resolution: float,
) -> dict:
    """Generate OmniLidar attributes for a solid-state multibeam-sonar fan.

    Emitters form a ``num_beams x num_elevation_rows`` grid: azimuth beam
    centers are evenly spaced across the fan (half-step inset at the edges so
    beam spacing is exactly ``az_span_deg / num_beams``), and each beam is
    repeated at every elevation row so the vertical fan is sampled. All
    emitters fire at the start of the scan (``fireTimeNs = 0``), and
    ``channelId`` encodes the 1-based azimuth beam index so returns can be
    regrouped by beam downstream.

    Args:
        num_beams: Number of azimuth beams across the fan.
        az_span_deg: Horizontal fan width in degrees (beams span +/- half).
        num_elevation_rows: Number of elevation rows sampling the vertical fan.
        el_span_deg: Vertical fan width in degrees (rows span +/- half).
        near_range: Minimum range in meters.
        far_range: Maximum range in meters.
        scan_rate_hz: Scan rate in Hz (also author as the sensor tick rate).
        range_resolution: Range resolution in meters.

    Returns:
        Attribute mapping for the ``Lidar`` authoring class, using the
        default ``s001`` emitter-state instance that the lidar core schema
        auto-applies.
    """
    beam_step = az_span_deg / num_beams
    azimuths = [(-az_span_deg / 2.0) + beam_step * (b + 0.5) for b in range(num_beams)]
    if num_elevation_rows == 1:
        elevations = [0.0]
    else:
        el_step = el_span_deg / (num_elevation_rows - 1)
        elevations = [(-el_span_deg / 2.0) + el_step * r for r in range(num_elevation_rows)]

    azimuth_deg = []
    elevation_deg = []
    channel_id = []
    for row_elevation in elevations:
        for b, beam_azimuth in enumerate(azimuths):
            azimuth_deg.append(beam_azimuth)
            elevation_deg.append(row_elevation)
            channel_id.append(b + 1)  # 1-based, matching schema convention
    num_emitters = len(azimuth_deg)

    return {
        "omni:sensor:Core:scanType": "SOLID_STATE",
        "omni:sensor:Core:numberOfEmitters": num_emitters,
        "omni:sensor:Core:emitterState:s001:azimuthDeg": azimuth_deg,
        "omni:sensor:Core:emitterState:s001:elevationDeg": elevation_deg,
        "omni:sensor:Core:emitterState:s001:channelId": channel_id,
        "omni:sensor:Core:emitterState:s001:fireTimeNs": [0] * num_emitters,
        "omni:sensor:Core:nearRangeM": near_range,
        "omni:sensor:Core:farRangeM": far_range,
        "omni:sensor:Core:scanRateBaseHz": scan_rate_hz,
        "omni:sensor:Core:rangeResolutionM": range_resolution,
        # spherical basic elements: each return is (azimuth deg, elevation
        # deg, range m) -- beam space, directly what a sonar reports
        "omni:sensor:Core:elementsCoordsType": "SPHERICAL",
    }


def sonar_scanline_from_returns(
    az_deg: np.ndarray,
    el_deg: np.ndarray,
    range_m: np.ndarray,
    num_beams: int,
    az_span_deg: float,
    el_span_deg: float,
    near_range: float,
    far_range: float,
) -> np.ndarray:
    """Collapse spherical lidar returns into a per-beam nearest-range scanline.

    Mimics a multibeam sonar's per-beam output: every return inside the
    vertical fan contributes to its azimuth beam, and each beam reports the
    nearest valid return (the first echo). Beams with no return are NaN.

    Args:
        az_deg: Return azimuths in degrees.
        el_deg: Return elevations in degrees.
        range_m: Return ranges in meters.
        num_beams: Number of azimuth beams across the fan.
        az_span_deg: Horizontal fan width in degrees.
        el_span_deg: Vertical fan width in degrees.
        near_range: Minimum valid range in meters.
        far_range: Maximum valid range in meters.

    Returns:
        Array of shape ``(num_beams,)`` with the nearest range per beam in
        meters (NaN where the beam saw nothing).
    """
    import numpy as np

    beam_step = az_span_deg / num_beams
    valid = (
        (range_m >= near_range)
        & (range_m <= far_range)
        & (np.abs(el_deg) <= el_span_deg / 2.0 + 1e-6)
        & (np.abs(az_deg) <= az_span_deg / 2.0 + 1e-6)
    )
    beam_index = np.clip(((az_deg + az_span_deg / 2.0) / beam_step).astype(np.int64), 0, num_beams - 1)
    scanline = np.full(num_beams, np.nan)
    for b, r in zip(beam_index[valid], range_m[valid]):
        if math.isnan(scanline[b]) or r < scanline[b]:
            scanline[b] = r
    return scanline


from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True})

output_dir = os.path.join(os.getcwd(), "_example_output_isaacsim.sensors.experimental.rtx", "create_lidar_multibeam_sonar")
os.makedirs(output_dir, exist_ok=True)

import numpy as np
import omni
import omni.replicator.core as rep
from isaacsim.core.experimental.objects import Cube
from isaacsim.sensors.experimental.rtx import Lidar, LidarSensor, parse_generic_model_output_data
from omni.replicator.core import Writer

# =============================================================================
# CREATE A SIMPLE SCENE
# =============================================================================
# same targets as create_acoustic_sonar_array.py so the outputs are comparable

for i, (x, y) in enumerate([(4.0, 0.0), (8.0, 3.0), (10.0, -5.0)]):
    Cube(f"/World/target_{i}", positions=np.array([x, y, 0.5]), scales=np.array([1.0, 1.0, 2.0]))

print("Created 3 target cubes")

# =============================================================================
# CREATE THE MULTIBEAM-FAN LIDAR
# =============================================================================

attributes = generate_multibeam_lidar_attributes(
    num_beams=args.num_beams,
    az_span_deg=args.az_span_deg,
    num_elevation_rows=args.num_elevation_rows,
    el_span_deg=args.el_span_deg,
    near_range=args.near_range,
    far_range=args.far_range,
    scan_rate_hz=args.scan_rate_hz,
    range_resolution=args.range_resolution,
)

lidar = Lidar(
    "/World/multibeam_sonar",
    tick_rate=float(args.scan_rate_hz),  # keep tickRate == scanRateBaseHz
    translations=np.array([0.0, 0.0, 0.5]),
    attributes=attributes,
)

sensor = LidarSensor(lidar, annotators=[])

print(
    f"Created multibeam-fan lidar at {lidar.paths[0]}: {args.num_beams} beams x "
    f"{args.num_elevation_rows} elevation rows over {args.az_span_deg} x {args.el_span_deg} deg, "
    f"{args.near_range}-{args.far_range} m at {args.scan_rate_hz} Hz"
    + (f", preset '{args.preset}'" if args.preset else "")
)


# =============================================================================
# WRITER: COLLAPSE RETURNS INTO THE PER-BEAM SONAR SCANLINE
# =============================================================================


class GmoMultibeamSonarWriter(Writer):
    """Writer that converts spherical lidar returns into a sonar range scanline."""

    def __init__(self) -> None:
        self.data_structure = "renderProduct"
        self.annotators = [rep.annotators.get("GenericModelOutput")]
        self._reported = False

    def write(self, data: dict[str, object]) -> None:
        """Build the per-beam scanline and report once."""
        if self._reported:
            # One-shot writer: skip BEFORE parsing -- parse_generic_model_output_data
            # copies and reparses the entire return buffer every frame otherwise.
            return
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

            # elementsCoordsType is SPHERICAL: x=azimuth deg, y=elevation deg, z=range m
            az = np.ctypeslib.as_array(gmo.x, shape=(n,))
            el = np.ctypeslib.as_array(gmo.y, shape=(n,))
            rng = np.ctypeslib.as_array(gmo.z, shape=(n,))
            scanline = sonar_scanline_from_returns(
                az, el, rng, args.num_beams, args.az_span_deg, args.el_span_deg, args.near_range, args.far_range
            )
            np.save(os.path.join(output_dir, "sonar_scanline_m.npy"), scanline)
            beam_step = args.az_span_deg / args.num_beams
            beam_angles = -args.az_span_deg / 2.0 + beam_step * (np.arange(args.num_beams) + 0.5)
            np.save(os.path.join(output_dir, "beam_angles_deg.npy"), beam_angles)

            hit = ~np.isnan(scanline)
            print(f"Returns: {n}; beams with echo: {int(hit.sum())}/{args.num_beams}")
            if hit.any():
                k = np.nanargmin(scanline)
                print(
                    f"  nearest echo: {scanline[k]:.3f} m at beam {beam_angles[k]:+.2f} deg; "
                    f"saved scanline to {output_dir}"
                )
            try:
                import matplotlib

                matplotlib.use("Agg")
                import matplotlib.pyplot as plt

                fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(8, 6))
                theta = np.radians(beam_angles[hit])
                ax.scatter(theta, scanline[hit], s=4)
                ax.set_thetalim(math.radians(-args.az_span_deg / 2), math.radians(args.az_span_deg / 2))
                ax.set_theta_zero_location("N")
                ax.set_rmax(args.far_range)
                ax.set_title("Multibeam-fan lidar: per-beam nearest range")
                fig.savefig(os.path.join(output_dir, "sonar_scanline.png"), dpi=120)
                print(f"  saved {os.path.join(output_dir, 'sonar_scanline.png')}")
            except ImportError:
                print("  matplotlib not available; skipped PNG rendering")


rep.WriterRegistry.register(GmoMultibeamSonarWriter)
sensor.attach_writer("GmoMultibeamSonarWriter")

# =============================================================================
# RUN SIMULATION
# =============================================================================
if args.test:
    stage = omni.usd.get_context().get_stage()
    stage.Export(os.path.join(output_dir, "stage.usda"))

timeline = omni.timeline.get_timeline_interface()
timeline.play()

frame_count = 0
while simulation_app.is_running() and (not args.test or frame_count < 40):
    simulation_app.update()
    frame_count += 1

timeline.stop()
sensor.destroy()
simulation_app.close()
