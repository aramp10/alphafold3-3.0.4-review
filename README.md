# AlphaFold3 v3.0.4 on the BU SCC: Review

**Status as of 2026-10-05:** installed and tested in `/share/pkg.8/alphafold3/3.0.4`. Linked in `/share/module.8/test` only, not published.

```bash
module use /share/module.8/test/
module help alphafold3/3.0.4
```

---

## Agenda

1. Approval to publish 3.0.4 system-wide (currently linked in `/share/module.8/test` for testing only).
2. Model weights: each user downloads their own, or one shared BU copy. `module help` MODEL WEIGHTS section points to the upstream download.
3. Note: once published, 3.0.4 becomes the default version (no `default`/`.modulerc` in `/share/module.8/math-eng/alphafold3/`).
4. Permissions: everything in `3.0.0/` and `3.0.1/` is now group-writable. The usual convention is a group-writable package folder with version folders that aren't, to guard against accidental deletion. Restore that with `chmod -R g-w` on the version folders?
5. Remove unused `/share/data/pkg.8/alphafold3/3.0.0/3.0.0_squashfs/alphafold_db_3.0.0.sqsh` (~258 GB)?

---

## Databases: no new download needed

3.0.4 uses the 3.0.0 databases in `/share/data/pkg.8/alphafold3/3.0.0/databases`.

| Check | Result |
|---|---|
| Download source | 3.0.0 `fetch_databases.py` and v3.0.1–v3.0.4 `fetch_databases.sh` all download the same 9 files from `https://storage.googleapis.com/alphafold-databases/v3.0`. "v3.0" is the database set for all v3.0.x; there is no 3.0.4 set. |
| Upstream changes | All 9 upstream `.zst` files: `Last-Modified` 2024-11-05, before the v3.0.0 release (2024-11-11). Rechecked 2026-10-05. |
| Default paths | 8 of 9 `${DB_DIR}/...` defaults in `run_alphafold.py` are unchanged. `--pdb_database_path` changed from `pdb_2022_09_28_mmcif_files.tar` (3.0.0) to unpacked `mmcif_files` (v3.0.1+). v3.0.4 still reads a `.tar` (`structure_stores.py`); `run_alphafold.sh` passes it. |
| Upstream statement | None on database compatibility in release notes v3.0.0–v3.0.4. |
| Real runs | Data pipeline exit 0, 4 PDB templates found; inference top ranking score 0.910 (L40S), 0.905 (V100). |

Not done: checksums of on-disk files (upstream md5 covers the `.zst` only).

---

## Changes vs 3.0.0

| Area | 3.0.0 | 3.0.4 |
|---|---|---|
| Container | Ubuntu 22.04, Python 3.11, 2.9 GB | Ubuntu 24.04, Python 3.12 + `uv`, 4.3 GB. spython conversion of upstream Dockerfile; edits marked `# SCC:` in `DIST/singularity.def` |
| Databases | `ALPHAFOLD_DB_DIR=/share/data/pkg.8/alphafold3/3.0.0/databases` | Same |
| Licenses | Code CC-BY-NC-SA 4.0 | Code Apache 2.0 (upstream change). Weights terms: only the code-license reference changed; prohibited-use policy: typo fix |
| Examples | — | Same, except version, a typo fix, and `2PV7` output folder (3.0.4 keeps the job name's case) |

### `run_alphafold.sh` (based on 3.0.0's)

- Adds `--pdb_database_path=/public_database/pdb_2022_09_28_mmcif_files.tar`.
- Detects user overrides written as `--flag` or `--flag=value`. User `--db_dir` drops both DB defaults (v3.0.4 appends a second `--db_dir` instead of replacing it).
- GPU compute capability < 8.0 (e.g. V100): also sets `XLA_FLAGS=--xla_disable_hlo_passes=custom-kernel-fusion-rewriter`; v3.0.4 refuses to run without it. Checks the job's own GPU, not GPU 0.
- Exit code passed through (`scc-singularity` always exits 0); arguments with spaces kept intact.
- `--help` works without `NSLOTS`; exits with a message if `ALPHAFOLD_DB_DIR` is unset.

### `modulefile.lua`

- `help()` rewritten, shorter; adds the qsub resource lines (`-pe omp 8` data pipeline; `-pe omp 4`, `gpus=1`, `gpu_c=6.0` inference).
- Dropped from 3.0.0's help: `gpu_memory=32G` and multi-GPU advice, JAX cache in `$TMPDIR` note, third-party guide link, "request weights individually".
- Commented-out squashfs line removed.

### Other

- 3.0.0 `install/bin/` has a stray `hs_err_pid3273985.log` (Apr 2025).

---

## Tests

Input: 2PV7 example from the 3.0.0 `examples/`.

- `test/test.qsub` (no GPU or weights): 5/5 passed, plain bash and qsub job 7876253.
- Data pipeline on the installed module: job 7876543, exit 0, 33 min (8 cores), 4 PDB templates. `2PV7_data.json` identical (md5) to the earlier test copy's.
- L40S inference on that output: job 7876544, exit 0, 123 s, top ranking score 0.910.
- V100 inference: tested on a test copy with the same `.sif` and `run_alphafold.sh` (md5 matched).

---

## Update 2026-10-08: after the review meeting

**Decisions:** publish after the edits and tests below. Model weights: each user downloads their own. 3.0.4 becomes the default. The unused `.sqsh` will be removed.

**Edits made**

| File | Change |
|---|---|
| `install/bin/run_alphafold.sh` | `--pdb_database_path` now points to the unpacked `/public_database/pdb_2022_09_28_mmcif_files/mmcif_files` instead of the `.tar`. |
| `install/bin/run_alphafold.sh` | Final call simplified to a plain `scc-singularity run`; the `--scc-preview`/`eval` code is removed. Argument pre-quoting is kept, so paths with spaces still work. The wrapper now always exits 0 (`scc-singularity` behavior). |
| `modulefile.lua`, `install/examples/af3_inference.qsub`, `test/runs/af3_inf.qsub` | Inference job: `-pe omp 4` → `-pe omp 8`. |
| `test/test.qsub` | Test 4 expects 10 database entries (unpacked directory added). Test 5 checks the error message only, not the exit code. |
| `notes.txt` | Wrapper description updated to match. |

`test/test.qsub` after the edits: 5/5 passed (plain bash).

**Test results (2026-10-08)**
- `test/test.qsub` batch job: 5/5 passed.
- Data pipeline, unpacked vs `.tar` (2PV7 example, 8 cores):

| | Unpacked | `.tar` | `.tar` (2026-10-05) |
|---|---|---|---|
| Template search step | 12 s | 509 s | 473 s |
| Whole data pipeline (chain A) | 1509 s | 2136 s | 1966 s |

  Output `2PV7_data.json` identical (md5) in all three. The MSA step (1493–1627 s) doesn't read the PDB files; its spread is run-to-run variation. Decision: keep the unpacked directory.
- L40S inference on the unpacked output: ranking score 0.91 (same as 2026-10-05).

**Still to do before publishing**
- Permissions fix on the unpacked `mmcif_files` (in progress).
- Inference on H200 and RTXP6000 (queued), reusing the existing data pipeline output.

**After publishing:** update the TechWeb AlphaFold3 page; point `module help` MODEL WEIGHTS to it.
