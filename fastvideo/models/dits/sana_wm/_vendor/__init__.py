# SPDX-License-Identifier: Apache-2.0
"""Vendored upstream code from
https://github.com/NVlabs/Sana (commit pinned in ``UPSTREAM.md`` once Phase 2 lands).

Phase 2A vendored:
- ``ops/frame_gdn/fused_recurrent_triton.py``  (kernel, no patches)
- ``ops/frame_gdn/scan_triton.py``             (kernel, no patches)

These are self-contained: their only third-party imports are ``torch``,
``triton``, and ``triton.language``. They are deliberately *not* re-exported
at the package level — the FastVideo wrapper in ``../model.py`` (Phase 2E)
will import them via their full sub-package path.
"""
