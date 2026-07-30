import os
import subprocess
import pytest

def test_reproducibility_execution_pipeline():
    # 1. Execute benchmark.py
    cmd_bench = ["python", "benchmark.py"]
    res_bench = subprocess.run(cmd_bench, capture_output=True, text=True)
    assert res_bench.returncode == 0
    assert "Results successfully saved to logs/benchmark_results.json." in res_bench.stdout
    
    # 2. Execute generate_tables.py
    cmd_tables = ["python", "generate_tables.py"]
    res_tables = subprocess.run(cmd_tables, capture_output=True, text=True)
    assert res_tables.returncode == 0
    assert "Wrote REPRODUCIBILITY_REPORT.md to workspace root." in res_tables.stdout
    
    # 3. Verify files exist
    assert os.path.exists("logs/benchmark_results.json")
    assert os.path.exists("REPRODUCIBILITY_REPORT.md")
    assert os.path.exists("logs/table_2.csv")
    assert os.path.exists("logs/table_16.csv")
    assert os.path.exists("logs/tables.tex")
