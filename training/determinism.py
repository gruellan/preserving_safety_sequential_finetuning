import logging
import os
import random

import numpy as np
import torch

logger = logging.getLogger(__name__)


def setup_determinism(seed: int = 42) -> None:
    """
    Configure torch and CUDA for deterministic train/eval runs
    call at start of each process before torch or model loaded or tensor allocated
    """
    # CUBLAS workspace must be set BEFORE CUDA init to have effect
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)

    torch.use_deterministic_algorithms(True, warn_only=True)

    logger.info(f"Determinism configured: seed={seed}")
