# ================================================================
# Created by: Li-Jungang
# Email: ljungang.02@gmail.com
# Description: utilities for prompt processing
# ================================================================
import re
import argparse
import json
import os
import subprocess
import math
import numpy as np
import torch

from typing import Dict, Tuple, List, Optional, Any
def extract_pred_letter(text: str, options: Dict[str, str]) -> str:
    """
    Robustly extract option letter (A/B/C/...) from model output.
    """
    if not text:
        return "Unknown"
    valid = set(k.upper() for k in options.keys())

    m = re.search(r"\b([A-Z])\b", text.upper())
    if m and m.group(1) in valid:
        return m.group(1)

    m = re.search(r"(OPTION|ANSWER)\s*[:\-]?\s*([A-Z])", text.upper())
    if m and m.group(2) in valid:
        return m.group(2)

    c0 = text.strip()[:1].upper()
    if c0 in valid:
        return c0

    return "Unknown"