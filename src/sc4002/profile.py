from pathlib import Path

import pandas as pd
from ydata_profiling import ProfileReport

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

train_df = pd.read_csv(DATA_DIR / "train.csv")
val_df = pd.read_csv(DATA_DIR / "dev.csv")
test_df = pd.read_csv(DATA_DIR / "test.csv")

train_profile = ProfileReport(train_df, title="Train Profiling Report")

train_profile.to_file(DATA_DIR / "train_report.html")

val_profile = ProfileReport(val_df, title="Validation Profiling Report")

val_profile.to_file(DATA_DIR / "val_report.html")

test_profile = ProfileReport(test_df, title="Test Profiling Report")

test_profile.to_file(DATA_DIR / "test_report.html")
