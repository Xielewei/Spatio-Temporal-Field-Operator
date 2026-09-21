"""Reproducible random seeds and portable runtime logging."""

import logging
import random
import sys
from pathlib import Path

import numpy as np
import torch

from utils.artifacts import PortableFormatter


def seed_anything(seed=42):
    """
    Step 1.2: Initialize random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def init_log(args):
    logger = logging.getLogger("stfo")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False
    formatter = PortableFormatter("%(asctime)s - %(message)s")
    for handler in [
        logging.FileHandler(Path(args.path) / "train.log"),
        logging.StreamHandler(sys.stdout),
    ]:
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    args.logger = logger
