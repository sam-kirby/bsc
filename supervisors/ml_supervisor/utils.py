import logging
import numpy as np
import sys

def init_logger() -> logging.Logger:
    logger = logging.getLogger("supervisor")
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter("%(levelname)s on %(thread)d at %(asctime)s: %(message)s")

    mh = logging.FileHandler("main.log")
    mh.setFormatter(formatter)
    mh.setLevel(logging.INFO)
    logger.addHandler(mh)

    soh = logging.StreamHandler(stream=sys.stdout)
    soh.setFormatter(formatter)
    soh.setLevel(logging.INFO)
    logger.addHandler(soh)

    dh = logging.FileHandler("debug.log")
    dh.setFormatter(formatter)
    logger.addHandler(dh)

    return logger

def pp_array(array: np.ndarray) -> str:
    return str([f"{x:.3e}" for x in array])
