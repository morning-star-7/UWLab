# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Run the Interactive World Simulator's ``main.py`` (unchanged) with RPM Lab runtime fixes. Used by ``train_iws.py``:

    python iws_main.py <iws repo> <hydra overrides...>

Fix: IWS's ``Attention`` restricts scaled-dot-product attention to the flash kernel on every GPU whose compute
capability is x.0 (meant for the A100; it also catches the H100) and lists no fallback. Flash attention takes fp16/bf16
only and IWS trains in fp32, so the dynamics model (stage 2/3) fails there with "No available kernel"; the A6000 (8.6)
takes IWS's other branch and never saw it. The memory-efficient and math kernels are added as fallbacks: flash is
still used whenever it applies.
"""

import os
import runpy
import sys

from torch.nn.attention import SDPBackend, sdpa_kernel

FALLBACKS = (SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH)


def sdpa_kernel_with_fallbacks(backends, *args, **kwargs):
    backends = [backends] if isinstance(backends, SDPBackend) else list(backends)
    return sdpa_kernel(backends + [b for b in FALLBACKS if b not in backends], *args, **kwargs)


def main():
    iws = os.path.abspath(sys.argv[1])
    sys.argv = [os.path.join(iws, "main.py"), *sys.argv[2:]]
    sys.path.insert(0, iws)  # as when main.py is run directly

    import interactive_world_sim.algorithms.models.attention as attention

    attention.sdpa_kernel = sdpa_kernel_with_fallbacks
    runpy.run_path(sys.argv[0], run_name="__main__")


if __name__ == "__main__":
    main()
