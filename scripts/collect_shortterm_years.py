from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd


DEFAULT_YEARS = list(range(2019, 2026))


def parse_mmdd(value: str) -> tuple[int, int]:
    try:
        month_text, day_text = value.split("-")
        month, day = int(month_text), int(day_text)
        datetime(2000, month, day)
