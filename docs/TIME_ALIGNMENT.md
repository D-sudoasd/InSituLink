# Frame–stress–strain time alignment

This documents the compatible `strainlink` entry. For the general InSituLink
request and multi-parameter pipeline, see [REQUEST.md](REQUEST.md).

This path places each diffraction frame on a tensile curve. It does not complete a missing stress or strain by interpolating a reference σ–ε curve. That reference-curve mapper remains `sxrd_stress_strain_mapper_gui_v3.py`.

There is no shared wall clock and no sub-second lock. The residual is about one measured frame interval.

## Clocks

Spectrum time comes from the finite per-frame detector start times. Exposure is the recorded count time. The mechanical query uses the exposure midpoint:

`t_mid = t_start + exposure / 2`

`t_mech = t_mid + Δt`

The inter-frame interval is `t_start[i] − t_start[i−1]` on those finite starts. These quantities are not the interval and are not Δt:

- nominal `frame_time`
- `detector_readout_time`
- `frame_index * exposure`
- a padded or virtual frame count
- the NeXus-GUI-minus-detector offset

On HEPS ID07 the count time is 1.0 s. Realized start-to-start intervals are about 1.41–1.56 s. Master-file `frame_time` is 1.0000001 s at room temperature and 2.0000001 s at 77 K. The pipeline flags that disagreement and still uses the measured starts.

## Δt

Δt is the tensile load-rise time minus the elastic lattice-strain onset time. Load rise is the force leaving preload. The onset is the first sustained departure of `ε_hkl` from its early baseline, inside the elastic window. It is not the macroscopic yield point.

Δt stays 0 and `onset_unresolved` is true when:

| Reason | Meaning |
| --- | --- |
| `onset_missing` | No sustained elastic departure |
| `onset_at_first_frame` | The series is already off the baseline at frame 0 |
| `rising_before_record` | Applying Δt would put already-rising strain before the tensile record |
| `subframe_disagreement` | The sub-frame threshold crossing disagrees by more than one measured frame interval |
| `xcorr_disagreement` | A sharp derivative correlation disagrees by more than one frame. A weak correlation does not move Δt |
| `load_rise_missing` | Force never holds a sustained rise above preload. A prefix spike shorter than the minimum run is not a load-rise, so no lag is invented |

Rejected candidate lags are stored as `rejected_dt_s`. The published `dt_s` is 0.

## Flags an agent must read

| Flag | What it means | What the pipeline does not do |
| --- | --- | --- |
| `unresolved_onset` | Δt was not measured | Does not invent a large lag |
| `pre_record_frames` | `t_mech` is before the tensile record | Stress, tensile strain, and force stay empty; `valid_intact` is 0 |
| `post_record_frames` | `t_mech` is after the tensile record ends | Same empty channels. The last tensile sample is not copied in |
| `duration_mismatch` | Photon span and tensile span differ by at least the profile threshold (30 s) | Does not stretch either clock |
| `nominal_period_disagrees` | Nominal `frame_time` differs from the measured start-to-start median | Does not replace the measured interval |
| `nonmonotonic_frame_clock` | A later finite start is not after the previous one | Does not reorder the frames |

`complex_cases` lists the flags that fired. `agent_notes` states the corresponding action. Analysis rows are those with `valid_intact = 1`.

## Stations

Both stations call the same core.

| Profile | Clock source |
| --- | --- |
| `heps_id07_eiger` | Finite Eiger `start_time` plus `count_time`. No HDF5 image read. A padded `start_time` array and decoy `frame_time`, readout, GUI offset, and virtual length are accepted and ignored as the clock |
| `extracted_frame_clock` | An already extracted table: `t_start_s`, `exposure_s`, `eps_hkl`. No HEPS HDF5 |

Run without the Tk GUI:

```powershell
python -m strainlink align --demo heps_id07_eiger
python -m strainlink align --demo extracted_frame_clock
```

The JSON object has `decision` and `table`.
