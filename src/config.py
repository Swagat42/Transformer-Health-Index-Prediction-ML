"""Central configuration for the Transformer Health Index project."""
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_DIR / "data" / "Health index1.csv"
RESULTS_DIR = ROOT_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
MODELS_DIR = ROOT_DIR / "models"

TARGET_COLUMN = "Health index"
DROP_COLUMNS = ["Life expectation"]  # removed to avoid target leakage

# 9 dissolved gases + 5 oil-quality parameters (column names as in the source CSV)
FEATURE_COLUMNS = [
    "Hydrogen", "Oxigen", "Nitrogen", "Methane", "CO", "CO2",
    "Ethylene", "Ethane", "Acethylene", "DBDS",
    "Power factor", "Interfacial V", "Dielectric rigidity", "Water content",
]

TEST_SIZE = 0.2
RANDOM_STATE = 42
CV_FOLDS = 5
