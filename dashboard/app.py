"""
FMCG Big Data ML Benchmark: Single-Node vs Distributed (PySpark)
Main Entry Point Wrapper (Routes to Overview)
"""

import os
import sys
import runpy

if __name__ == "__main__":
    overview_path = os.path.join(os.path.dirname(__file__), "Overview.py")
    runpy.run_path(overview_path, run_name="__main__")
