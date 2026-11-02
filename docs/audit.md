# Project audit — 2 November 2026

## Scope and changes

Reviewed every tracked Python script, both standalone Hugging Face deploy folders, dataset configuration, dependency files, publishing behavior, documentation, and the existing GitHub release. The original repository had no GitHub Actions workflows or automated tests.

Fixed:

- **Space startup crash:** `/app` has no second parent; resolve the sibling model path without indexing ancestors. CI now imports the standalone app from `/app`.
- **Model concurrency:** photo, video and live events share one queued execution slot; a lock also protects direct calls to the shared model.
- **Video memory and storage:** consume results incrementally, enforce 30-second clips, limit uploads to 25 MB, remove failed outputs, expire served cache and clean old source videos on subsequent video requests.
- **Model configuration:** explicit Hub model selection overrides old bundled weights; explicit missing paths fail clearly.
- **UI behavior:** clear photo and result together; improve mobile spacing and heading scale; explain server processing and the detector's actual scope.
- **Data preparation:** validate the 21-class order and every box before remapping, retain original labels/metadata, and require completed remapping before training/export.
- **Dataset downloads:** use the documented Roboflow export API without the SDK's conflicting OpenCV dependency, stage/validate downloads before replacing data, reject archive traversal, preserve original data on failure, and avoid printing credential-bearing download URLs.
- **TFLite export:** use standard INT8 TFLite, select the integer-I/O artifact, and check both input/output tensor types and scales. The old script exported Edge TPU binaries under a generic TFLite name.
- **Desktop scripts:** release cameras/windows even if inference raises an error.
- **Publishing:** replace force-pushed orphan histories with ordinary Hub commits, protect against concurrent edits, validate destinations, exclude secrets/caches recursively, and make dry runs read-only.
- **Dependencies:** CPU-compatible PyTorch defaults; remove the oversized transitive dependency list and duplicate headless OpenCV installation from declared requirements.
- **Automation:** add checks, explicit manual Hub deployment, and manual maintenance of the existing checkpoint release.
- **Documentation:** replace the remote stock header with a local banner and real website preview; document setup, model semantics, limitations, workflow requirements, release checksums, and useful Hub links.

## Verification

- Ruff lint and Python compilation.
- Fourteen automated tests, including real checkpoint photo/live-frame inference, an actual encoded video, bad-label preservation, full-export prevalidation, shared queues, model precedence, standalone `/app` paths, non-Hub destination rejection and publishing safeguards.
- A local Gradio HTTP photo request returned annotated output and two detections.
- Browser rendering at desktop and mobile widths; no horizontal overflow or browser errors in the checked screens. Better Design guidance/review tools were attempted but unavailable because the account's monthly allowance was exhausted; no Better Design score is claimed.
- Linux Python 3.11 dependency resolution.
- Existing release `best.pt` downloaded and SHA-256 matched against the bundled checkpoint: `f300d94325772555e8e22ffe6f88755fe9c696d75e4d4494dbbe2cfc0debf937`. Original checkpoint and tag preserved; release notes and checksum asset updated.
- GitHub checks, manual release maintenance and manual Hub publishing workflows all completed successfully. The repository's encrypted `HF_TOKEN` Actions secret was configured for Hub publishing.
- The updated Space reached RUNNING; real production photo, video and live-frame endpoints returned annotated results. The Python client's live-stream status parser could not handle `process_streaming`, so that endpoint was verified through Gradio's HTTP/SSE API instead. Physical browser-camera streaming remains untested.
- Created a local `.env` with available Hub credentials and IDs and encrypted it using the owner's `agel` helper. The Roboflow key was unavailable. The private identity configured by `ageu` was absent, so decryption round-trip verification could not be performed.

## Limits

No fresh model training, private Roboflow download, held-out accuracy evaluation, full TensorFlow conversion, physical webcam/device test or ATM field validation was performed. The repo contains no reproducible benchmark report or top-level license grant. Public deployment and Actions outcomes must be checked against their current runtime/run status; an upload or push alone does not prove readiness. Manual Hub publishing in Actions requires an `HF_TOKEN` repository secret; local cached credentials do not configure that secret.
