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

"""Dense multibeam-sonar-style acoustic array on a SINGLE OmniAcoustic prim.

This example builds a dense multi-element acoustic array to approximate an
imaging (multibeam) sonar, and beamforms the per-element output into the
polar beams-x-range image such a sonar produces. The key design decision is
that all elements live on ONE OmniAcoustic prim as ``sensorMount``
multi-apply schema instances -- NOT as separate AcousticSensor objects:

- One prim means ONE render product / Hydra pipeline, instead of one per
  element. Per-render-product overhead is the dominant fixed VRAM/perf cost
  of an RTX sensor, so N separate sensors would multiply that cost N-fold
  for no modeling benefit.
- The WPM acoustic model natively simulates multi-transmitter /
  multi-receiver topologies inside one trace pass: ray cost scales with the
  number of *firing* transmitters, while additional receivers are nearly
  free. The GMO output labels every amplitude sample with (x=tx mount ID,
  y=rx mount ID, z=channel ID), giving the per-element waveform matrix that
  the beamformer consumes.

The ``--preset oculus-m3000d-lf`` configuration approximates the Blueprint
Subsea Oculus M3000d multibeam sonar in its low-frequency mode
(1.2 MHz, 130 deg x 20 deg aperture, 512 beams at 0.25 deg separation,
0.6 deg angular resolution, 0.1-30 m range):

- 168 receive elements at half-wavelength spacing give the same aperture
  measured in wavelengths (~84 lambda) as the real 105 mm array, so the
  beam pattern (0.6 deg beams, no grating lobes across the +/-65 deg fan)
  is preserved even though the in-air wavelength differs.
- One transmitter fires a fan ping per frame; all elements receive; 512
  beams are synthesized in post by delay-and-sum, exactly like the real
  unit beamforms its stave data.

Fidelity notes (be aware of what this model can and cannot reproduce):

- The WPM acoustic model is an IN-AIR ultrasonic model (c = 343 m/s; the
  schema has no speed-of-sound parameter). Geometry in wavelengths (beam
  pattern, FOV) is preserved, but time-of-flight is ~4.4x longer than in
  water: a 30 m listening window takes 175 ms, capping the update rate at
  ~5 Hz instead of the real unit's range-dependent up-to-40 Hz.
- GMO ``scalar`` samples are amplitude ENVELOPES (the default sample
  duration spans several carrier periods), so beamforming here is
  non-coherent (envelope delay-and-sum). Peak/echo localization is
  accurate to a fraction of a degree, but the -3 dB beamwidth is set by
  time-delay alignment of envelopes (``~2.6 * c * pulse_sigma /
  aperture``) rather than the coherent ``0.88 * lambda / aperture``: ~5
  deg with this preset instead of the real unit's 0.6 deg. This floor is
  frequency-invariant (the minimum pulse of a few carrier periods scales
  with 1/f exactly as the wavelength-sized aperture does), so the only
  lever is more aperture wavelengths: doubling the element count roughly
  halves the beamwidth. True 0.6 deg beams would require phase-coherent
  element data, which the GMO envelope output does not carry -- if you
  need per-beam ranges at the real unit's angular resolution, pair this
  sensor with (or substitute) an RTX lidar configured as a 130 x 20 deg,
  512-column fan, which yields idealized multibeam geometry directly.
  Speckle and the real unit's 2.5 mm bandwidth-driven range resolution
  are likewise not reproduced (envelope range resolution here: ~0.9 cm).
- Directivity/gain polynomials are the schema's generic ultrasonic
  transducer defaults, not the Oculus element response.
"""

from __future__ import annotations

import argparse
import math
import os

SPEED_OF_SOUND = 343.0  # m/s (air; the WPM acoustic model is an in-air ultrasonic model)

# Preset approximating the Oculus M3000d low-frequency mode. Element count:
# a 0.6 deg beam needs an ~84-lambda aperture (0.88 * lambda / theta), which
# at half-wavelength spacing is 168 elements; lambda/2 spacing is required to
# steer across the full +/-65 deg fan without grating lobes.
PRESETS = {
    "oculus-m3000d-lf": dict(
        num_elements=168,
        num_transmitters=1,
        geometry="line",
        center_frequency=51200.0,  # in-air carrier; aperture is sized in wavelengths
        az_span_deg=130.0,
        el_span_deg=20.0,
        tick_rate=5.0,  # 30 m two-way ToF in air is 175 ms -> at most ~5.7 Hz
        sample_duration=25.6e-6,  # 4.4 mm range bins (two-way)
        pulse_duration=51.2e-6,  # ~0.9 cm envelope range resolution; 2.6 carrier cycles
        bandwidth=19500.0,  # ~1 / pulse_duration
        num_beams=512,
    ),
}

# two-stage parsing so a preset provides defaults while explicit flags win
_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--preset", type=str, default=None, choices=sorted(PRESETS.keys()))
_pre_args, _ = _pre.parse_known_args()

parser = argparse.ArgumentParser(description="Dense multibeam-sonar-style RTX Acoustic array example.", parents=[_pre])
parser.add_argument("--test", default=False, action="store_true", help="Run in test mode.")
parser.add_argument("--num-elements", type=int, default=128, help="Number of transducer elements in the array.")
parser.add_argument(
    "--num-transmitters",
    type=int,
    default=1,
    help="Number of elements that fire (spread evenly across the array); all elements receive.",
)
parser.add_argument("--geometry", type=str, default="grid", choices=["line", "grid", "ring"], help="Element layout.")
parser.add_argument(
    "--center-frequency", type=float, default=51200.0, help="Center frequency in Hz (sets lambda/2 element spacing)."
)
parser.add_argument("--az-span-deg", type=float, default=None, help="Horizontal (azimuth) aperture in degrees.")
parser.add_argument("--el-span-deg", type=float, default=None, help="Vertical (elevation) aperture in degrees.")
parser.add_argument("--tick-rate", type=float, default=20.0, help="Sensor tick rate in Hz.")
parser.add_argument("--sample-duration", type=float, default=None, help="Waveform sample duration in seconds.")
parser.add_argument("--pulse-duration", type=float, default=None, help="Pulse duration in seconds.")
parser.add_argument("--bandwidth", type=float, default=None, help="Bandwidth in Hz.")
parser.add_argument("--num-beams", type=int, default=256, help="Number of beams to synthesize by delay-and-sum.")
parser.add_argument(
    "--fire-interval-ms",
    type=float,
    default=2.0,
    help="Time offset between consecutive transmitter firings within a frame (ms).",
)
if _pre_args.preset is not None:
    parser.set_defaults(**PRESETS[_pre_args.preset])
args, _ = parser.parse_known_args()


def generate_sonar_array_attributes(
    num_elements: int,
    num_transmitters: int,
    geometry: str,
    center_frequency: float,
    fire_interval_ms: float,
    az_span_deg: float | None = None,
    el_span_deg: float | None = None,
    sample_duration: float | None = None,
    pulse_duration: float | None = None,
    bandwidth: float | None = None,
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
        az_span_deg: Horizontal aperture in degrees (None: schema default).
        el_span_deg: Vertical aperture in degrees (None: schema default).
        sample_duration: Waveform sampling in seconds (None: schema default).
        pulse_duration: Pulse duration in seconds (None: schema default).
        bandwidth: Bandwidth in Hz (None: schema default).

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
    for name, value in (
        ("azSpanDeg", az_span_deg),
        ("elSpanDeg", el_span_deg),
        ("sampleDuration", sample_duration),
        ("pulseDuration", pulse_duration),
        ("bandwidth", bandwidth),
    ):
        if value is not None:
            attributes[f"omni:sensor:WpmAcoustic:{name}"] = value
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


def beamform_das(
    waveforms: np.ndarray,
    rx_y: np.ndarray,
    sample_duration: float,
    num_beams: int,
    az_span_deg: float,
    speed_of_sound: float = SPEED_OF_SOUND,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Non-coherent (envelope) delay-and-sum beamformer for a line array.

    Synthesizes the polar beams-x-range image a multibeam sonar outputs from
    the per-element envelope waveforms. Uses focused (near-field, spherical)
    delays: for a candidate scatterer at range ``r`` along beam direction
    ``phi``, the expected two-way time at element ``e`` is
    ``(r + |target - p_e|) / c`` (transmit from array center, receive at the
    element), which handles both near- and far-field geometry.

    Args:
        waveforms: Envelope matrix of shape ``(num_elements, num_samples)``,
            row ``e`` being the waveform of the element at ``rx_y[e]``.
        rx_y: Element positions along the array axis, in meters.
        sample_duration: Time per waveform sample, in seconds.
        num_beams: Number of beams across the fan.
        az_span_deg: Total fan width in degrees (beams span +/- half of it).
        speed_of_sound: Propagation speed in m/s.

    Returns:
        ``(image, beam_angles_deg, ranges_m)`` where ``image`` has shape
        ``(num_beams, num_samples)``.

    Note:
        This is a plain-NumPy reference implementation that takes on the
        order of seconds per 512-beam image (the example's writer runs it
        once, on the first valid frame). For per-frame imaging, port the
        gather-and-sum to Warp/CUDA.
    """
    import numpy as np

    num_elements, num_samples = waveforms.shape
    beam_angles = np.linspace(-az_span_deg / 2.0, az_span_deg / 2.0, num_beams)
    # two-way range axis: sample k corresponds to a scatterer at r = k*c*dt/2
    ranges = np.arange(num_samples) * speed_of_sound * sample_duration / 2.0
    image = np.zeros((num_beams, num_samples), dtype=np.float64)
    element_indices = np.arange(num_elements)[:, None]
    for b, angle_deg in enumerate(beam_angles):
        angle = math.radians(angle_deg)
        # candidate scatterer positions along this beam (boresight = +X, array along Y)
        tx_x = ranges * math.cos(angle)
        tx_y = ranges * math.sin(angle)
        # per-element receive distance, shape (num_elements, num_samples)
        dist_rx = np.sqrt(tx_x[None, :] ** 2 + (tx_y[None, :] - rx_y[:, None]) ** 2)
        t = (ranges[None, :] + dist_rx) / speed_of_sound
        idx = np.rint(t / sample_duration).astype(np.int64)
        valid = idx < num_samples
        np.clip(idx, 0, num_samples - 1, out=idx)
        image[b] = np.sum(waveforms[element_indices, idx] * valid, axis=0)
    return image, beam_angles, ranges


from isaacsim import SimulationApp

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
# targets spread across the fan (boresight +X): ~0 deg, ~+21 deg, ~-27 deg

for i, (x, y) in enumerate([(4.0, 0.0), (8.0, 3.0), (10.0, -5.0)]):
    Cube(f"/World/target_{i}", positions=np.array([x, y, 0.5]), scales=np.array([1.0, 1.0, 2.0]))

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
    az_span_deg=args.az_span_deg,
    el_span_deg=args.el_span_deg,
    sample_duration=args.sample_duration,
    pulse_duration=args.pulse_duration,
    bandwidth=args.bandwidth,
)
element_positions = [
    attributes[f"omni:sensor:WpmAcoustic:sensorMount:m{i + 1:03d}:position"] for i in range(args.num_elements)
]

acoustic = Acoustic(
    "/World/sonar_array",
    aux_output_level="BASIC",  # numSgws / numSamplesPerSgw in auxiliary data
    tick_rate=args.tick_rate,
    translations=np.array([0.0, 0.0, 0.5]),
    attributes=attributes,
)

sensor = AcousticSensor(acoustic, annotators=[])

spacing_mm = SPEED_OF_SOUND / args.center_frequency / 2.0 * 1e3
print(
    f"Created sonar array at {acoustic.paths[0]}: {args.num_elements} elements "
    f"({args.geometry}, {spacing_mm:.2f} mm spacing), {args.num_transmitters} transmitter(s), "
    f"tick rate {args.tick_rate} Hz"
    + (f", preset '{args.preset}'" if args.preset else "")
)


# =============================================================================
# WRITER: ASSEMBLE THE WAVEFORM MATRIX AND BEAMFORM THE SONAR IMAGE
# =============================================================================


class GmoSonarArrayWriter(Writer):
    """Writer that beamforms GMO element waveforms into a beams-x-range image."""

    def __init__(self) -> None:
        self.data_structure = "renderProduct"
        self.annotators = [rep.annotators.get("GenericModelOutput")]
        self._reported = False

    def write(self, data: dict[str, object]) -> None:
        """Build the per-element waveform matrix, beamform, and report once."""
        if self._reported:
            # One-shot writer: skip BEFORE parsing -- parse_generic_model_output_data
            # copies and reparses the entire waveform buffer, a per-frame tax that
            # would otherwise continue for the whole (unbounded) run.
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

            samples_per_sgw = getattr(gmo, "numSamplesPerSgw", 0)
            tx_ids = np.ctypeslib.as_array(gmo.x, shape=(n,))
            rx_ids = np.ctypeslib.as_array(gmo.y, shape=(n,))
            if not samples_per_sgw:
                # without auxiliary data, derive the count from the first
                # signal-way boundary (samples of one signal way are contiguous)
                boundary = np.flatnonzero((np.diff(tx_ids) != 0) | (np.diff(rx_ids) != 0))
                samples_per_sgw = int(boundary[0] + 1) if boundary.size else n
            num_sgws = n // samples_per_sgw

            amplitudes = np.ctypeslib.as_array(gmo.scalar, shape=(n,))
            waveforms = amplitudes[: num_sgws * samples_per_sgw].reshape(num_sgws, samples_per_sgw)
            sgw_rx = rx_ids[::samples_per_sgw][:num_sgws]
            sgw_tx = tx_ids[::samples_per_sgw][:num_sgws]
            print(f"Waveform matrix: {waveforms.shape} (signal ways x samples)")

            # beamform one ping: with multiple transmitters there are several
            # signal ways per receiver, so keep only the first transmitter's
            # rows (one row per receiving element)
            first_tx_mask = sgw_tx == sgw_tx.min()
            waveforms = waveforms[first_tx_mask]
            sgw_rx = sgw_rx[first_tx_mask]

            # order rows by element index. GMO reports mount IDs, which are
            # 0-based element indices (see the receiverIndices authoring above),
            # so each row's Y coordinate is looked up by its OWN mount ID --
            # never by row position: when some elements didn't receive, indexing
            # positions by row counter would beamform surviving elements at the
            # wrong coordinates and silently corrupt the array geometry.
            order = np.argsort(sgw_rx, kind="stable")
            waveforms = waveforms[order]
            row_ids = sgw_rx[order].astype(int)
            valid = (row_ids >= 0) & (row_ids < len(element_positions))
            if not valid.all():
                print(f"  WARNING: dropping {int((~valid).sum())} signal ways with out-of-range receiver IDs")
                waveforms = waveforms[valid]
                row_ids = row_ids[valid]
            if row_ids.size != len(element_positions):
                print(
                    f"  WARNING: {row_ids.size} receivers seen, expected {len(element_positions)}; "
                    "beamforming with the receivers present"
                )
            positions_y = np.asarray([p[1] for p in element_positions])
            rx_y = positions_y[row_ids]
            sample_duration = args.sample_duration or 0.0001024  # schema default
            az_span = args.az_span_deg or 90.0  # schema default

            image, beam_angles, ranges = beamform_das(
                waveforms, rx_y, sample_duration, args.num_beams, az_span
            )
            np.save(os.path.join(output_dir, "sonar_image.npy"), image)
            np.save(os.path.join(output_dir, "beam_angles_deg.npy"), beam_angles)
            np.save(os.path.join(output_dir, "ranges_m.npy"), ranges)
            peak = np.unravel_index(np.argmax(image), image.shape)
            print(
                f"Sonar image: {image.shape} (beams x range bins), saved to {output_dir}\n"
                f"  strongest echo: beam {beam_angles[peak[0]]:+.1f} deg, range {ranges[peak[1]]:.2f} m"
            )
            try:
                import matplotlib

                matplotlib.use("Agg")
                import matplotlib.pyplot as plt

                db = 20.0 * np.log10(np.maximum(image, 1e-12) / max(image.max(), 1e-12))
                fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(8, 6))
                theta = np.radians(beam_angles)
                ax.pcolormesh(theta, ranges, np.clip(db, -40.0, 0.0).T, cmap="viridis")
                ax.set_thetalim(theta.min(), theta.max())
                ax.set_theta_zero_location("N")
                ax.set_title("Acoustic array: beamformed sonar image (dB)")
                fig.savefig(os.path.join(output_dir, "sonar_image.png"), dpi=120)
                print(f"  saved {os.path.join(output_dir, 'sonar_image.png')}")
            except ImportError:
                print("  matplotlib not available; skipped PNG rendering")


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
while simulation_app.is_running() and (not args.test or frame_count < 40):
    simulation_app.update()
    frame_count += 1

timeline.stop()
sensor.destroy()
simulation_app.close()
