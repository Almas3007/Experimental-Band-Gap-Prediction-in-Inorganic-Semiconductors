"""Rebuild the JARVIS-DFT tables used by the study scripts from the raw JARVIS zip files in data_raw/.

The raw files (d2-12-12-2022.json.zip, jdft_3d-9-24-2025.json.zip) are the unmodified JARVIS-DFT
releases from figshare. If they are missing, jarvis-tools downloads them.

Usage:  python common/prepare_jarvis.py [2d] [3d]      (default: both available)
"""
import os
import sys

import pandas as pd
from jarvis.db.figshare import data

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data_raw")
os.makedirs(RAW, exist_ok=True)

want = sys.argv[1:] or [k for k, f in (("2d", "d2-12-12-2022.json.zip"), ("3d", "jdft_3d-9-24-2025.json.zip"))
                        if os.path.exists(os.path.join(RAW, f))] or ["2d", "3d"]

if "2d" in want:
    df = pd.DataFrame(data("dft_2d", store_dir=RAW))
    df.to_pickle(os.path.join(RAW, "jarvis_dft_2d_full.pkl"))
    print("dft_2d:", df.shape, "-> data_raw/jarvis_dft_2d_full.pkl")
if "3d" in want:
    df = pd.DataFrame(data("dft_3d", store_dir=RAW)).drop(columns=["atoms"], errors="ignore")
    df.to_pickle(os.path.join(RAW, "jarvis_dft_3d_noatoms.pkl"))
    print("dft_3d:", df.shape, "-> data_raw/jarvis_dft_3d_noatoms.pkl")
