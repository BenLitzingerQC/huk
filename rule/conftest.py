import sys
from unittest.mock import MagicMock

for mod in [
    "hydra",
    "mlflow",
    "da_hf5_dz",
    "da_hf5_dz.config",
    "da_hf5_dz.configs",
    "da_hf5_dz.configs.plotting",
    "da_hf5_dz.configs.plotting.plotting",
    "da_hf5_utils",
    "da_hf5_utils.db2",
    "evaluate_rule",
    "sqlalchemy",
    "sqlalchemy.types",
    "matplotlib",
    "matplotlib.patches",
    "matplotlib.pyplot",
    "seaborn",
]:
    sys.modules.setdefault(mod, MagicMock())
