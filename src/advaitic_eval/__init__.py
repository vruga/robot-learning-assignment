"""Fixed environment and evaluator for the Advaitic plate-transfer assignment."""

import os

# Must be set before the ONNX library is first imported. Evaluation needs
# local controller inference only, not optional runtime telemetry.
os.environ["ORT_DISABLE_TELEMETRY"] = "1"

__version__ = "0.1.0"
