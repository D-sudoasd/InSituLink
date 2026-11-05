# InSituLink agent entry

G13 reference-model follow-up: read [the approved stress-to-strain model record](docs/HEPS_20261027_G13_REFERENCE_STRAIN_MODEL.md). The new estimated-total-strain fields are separate from the unrecovered raw channel. Preserve original stress/clocks and use branch/status/model-spread metadata; this model uses no crosshead or compliance correction.


TC4 five-sample follow-up: see [the per-file audit and G21 area decision](docs/HEPS_20261027_TC4_FIVE_COLUMN_AUDIT.md). G9 is prepeak-only; G13 full strain remains unavailable. User requested G21 primary stress from field area 1.9872 mm²; keep TXT stress separately and keep the direct strain unchanged. The per-file sidecars and candidate statuses apply; no global Tension2010 unit override.


Read [README.md](README.md) for purpose and [docs/REQUEST.md](docs/REQUEST.md) for the version-1 request and output contract. Legacy `strainlink` and the reference-curve GUI remain compatible independent tools.

Known tensile-export case: before processing this HEPS campaign's `G14.txt` / valid scan `G14-1`, read [the column-semantics case record](docs/HEPS_20261027_G14_TENSION2010_COLUMN_SEMANTICS.md) and the TXT's sidecar notes. The user accepted column 3's relative values as strain percent on 2026-11-04; this is a working interpretation, not independent instrument calibration. Do not divide that signal by a gauge length or multiply it by 100 again. Do not generalize this rule to other files or confirm the frame clock from it.

For ambiguous tensile exports, check numerical column relationships and dimensional consistency before trusting unit labels, deriving strain, or fitting machine compliance. Preserve the raw columns and distinguish a unit interpretation from sensor calibration and clock evidence.

1. Start with experiment metadata and external records. Declare columns, units, time origins and actual clock evidence; never infer geometry or synchronization from filenames.
   Use `python -m insitulink prepare --frames PATH --source ID=PATH --output NEW_DIRECTORY` to generate an editable request and field-specific actions. Reuse an explicit profile when appropriate; imported registration IDs are not evidence of synchronization.
2. Keep raw inputs intact. Write an experiment request and use a separate output directory. Run `python -m insitulink run --request REQUEST --output OUTPUT`; use its JSON response before reading additional artifacts.
3. Independent devices need independent clock relationships. A confirmed shared clock includes a common origin. A thermal waveform delay alone is a candidate. Unresolved offsets are null in the new interface.
4. Use spectral/image preanalysis only when needed. Preserve physical frame identity, unsupported detector bins and invalid fit flags. Do not force three peaks, assign phases/hkl, or remove actual sample-response delay to improve curve agreement.
5. For `needs_input`, inspect the specific issues, decisions and report, then ask for missing evidence. Modify the request only using supplied or independently established facts and resume. Never automatically confirm an unverified candidate.
6. Deliver the complete alignment output with `handoff.json`, keeping the original request, frame metadata, external logs and synchronization evidence accessible for source-hash checks. InSituLink publishes a file contract and does not launch its consumer. In DIDASFlow's `didasflow/` directory, `python run_cli.py handoff --handoff ALIGN/handoff.json --profile profile.json --output RESULTS` performs the supported Full 1D Q integration with an independently supplied DIDASFlow calibration profile. Its initial scope is whole 2D frames from rank-2/rank-3 HDF5 datasets; other products and custom workflows are unsupported. Require the consumer's completed per-frame bindings before claiming that integration finished.
   Include `frame_records.json` for physical identity, local exposure coordinates and parameter support rows. Preserve acquisition/asset IDs when relocating inputs; keep raw-address verification separate from file existence and pixel-content verification.
   Retain all records and the original selected-frame order, including selected frames with invalid or missing parameters. Use the consumer's explicit `--path-map` for relocation and a new output or `--existing-output versioned` when source/calibration identity changes. `completed` and `ready_for_integration` do not replace consumer preflight. See [DIDASFlow handoff instructions](https://github.com/D-sudoasd/DIDASFlow/blob/main/didasflow/docs/insitulink-handoff.md).
7. `inspect` checks output artifacts; `resume` revalidates inputs. Use an already completed command's status instead of immediately inspecting again. Synthetic validation is separate from beamline acceptance.

Developer checks: `py -3 -m unittest discover -s tests -v`. Run focused tests while iterating, then the full suite for shared/public-interface changes. Do not alter DIDASFlow in this repository task.
