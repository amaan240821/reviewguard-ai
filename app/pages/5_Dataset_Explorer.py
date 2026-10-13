import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from app.ui.views import dataset_explorer

dataset_explorer()
