# InSituLink agent entry

Read [README.md](README.md) for purpose and [docs/REQUEST.md](docs/REQUEST.md) for the version-1 request and output contract. Legacy `strainlink` and the reference-curve GUI remain compatible independent tools.

1. Start with experiment metadata and external records. Declare columns, units, time origins and actual clock evidence; never infer geometry or synchronization from filenames.
   Use `python -m insitulink prepare --frames PATH --source ID=PATH --output NEW_DIRECTORY` to generate an editable request and field-specific actions. Reuse an explicit profile when appropriate; imported registration IDs are not evidence of synchronization.
2. Keep raw inputs intact. Write an experiment request and use a separate output directory. Run `python -m insitulink run --request REQUEST --output OUTPUT`; use its JSON response before reading additional artifacts.
3. Independent devices need independent clock relationships. A confirmed shared clock includes a common origin. A thermal waveform delay alone is a candidate. Unresolved offsets are null in the new interface.
4. Use spectral/image preanalysis only when needed. Preserve physical frame identity, unsupported detector bins and invalid fit flags. Do not force three peaks, assign phases/hkl, or remove actual sample-response delay to improve curve agreement.
5. For `needs_input`, inspect the specific issues, decisions and report, then ask for missing evidence. Modify the request only using supplied or independently established facts and resume. Never automatically confirm an unverified candidate.
6. Deliver the parameter table, physical-frame manifest and clock evidence with `handoff.json`. This file contract is for downstream integration; DIDASFlow does not yet automatically consume it. Do not claim production integration or combine frames.
   Include `frame_records.json` for physical identity, local exposure coordinates and parameter support rows. Preserve acquisition/asset IDs when relocating inputs; keep raw-address verification separate from file existence and pixel-content verification.
7. `inspect` checks output artifacts; `resume` revalidates inputs. Use an already completed command's status instead of immediately inspecting again. Synthetic validation is separate from beamline acceptance.

Developer checks: `py -3 -m unittest discover -s tests -v`. Run focused tests while iterating, then the full suite for shared/public-interface changes. Do not alter DIDASFlow in this repository task.
