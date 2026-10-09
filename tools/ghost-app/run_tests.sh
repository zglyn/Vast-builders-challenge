#!/usr/bin/env bash
# Run GHOST unit tests (what-if + Phase-1 geometric video edit).
set -euo pipefail
cd "$(dirname "$0")"
echo "ffmpeg: $(command -v ffmpeg || echo MISSING — MP4 export tests will skip/fail)"
python3 -m unittest \
  tests.test_whatif \
  tests.test_video_edit \
  tests.test_sprint_quality \
  tests.test_phase2_refine \
  -v
echo
echo "OK — all tests passed."
echo "Manual UI check: workshop App → What-If Lab → Parse → Run ensemble → Export"
echo "  (Phase-2 Poisson refine on; leave synthetic unchecked)."
echo "Note: local nvidia.com/gpu Jobs cannot schedule here; optional GHOST_GEN_INPAINT_URL."
