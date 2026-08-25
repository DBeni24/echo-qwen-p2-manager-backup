import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SCRIPTS = ROOT / "scripts"

sys.path.insert(0, str(SCRIPTS))

import generate_manager_policy_dataset_v05_safe_v21 as runtime


model_path = os.environ.get("ECHO_MODEL_PATH")
run_root = os.environ.get("ECHO_RUN_ROOT")


if model_path:
    runtime.base.MODEL = Path(
        model_path
    ).expanduser().resolve()


if run_root:
    runtime.base.DEFAULT_RUN_ROOT = Path(
        run_root
    ).expanduser().resolve()


print(
    "ECHO_MANAGER_RUNTIME="
    + runtime.VERSION,
    flush=True,
)

print(
    "ECHO_MODEL_PATH="
    + str(runtime.base.MODEL),
    flush=True,
)

print(
    "ECHO_RUN_ROOT="
    + str(runtime.base.DEFAULT_RUN_ROOT),
    flush=True,
)


runtime.main()
