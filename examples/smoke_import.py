"""Import-check for the BC-NO Colab pack (no GPU / no training)."""
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(ROOT))

for name in [
    "bcno_model_2d",
    "bcno_model_3d",
    "bcno_elasticity_2d",
    "baselines_2d",
    "colab_data_gen",
    "rve_discrete_operator_2d",
]:
    spec = importlib.util.find_spec(name)
    if spec is None:
        raise SystemExit(f"missing module: {name}")
    print("ok", name)
print("smoke_import passed")
