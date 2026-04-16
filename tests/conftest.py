import pytest
import numpy as np
from unittest.mock import MagicMock, patch
import sys
import os

# Parchamos TODO antes de que main.py se importe
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'api'))

mock_tensor = MagicMock()
mock_tensor.numpy.return_value = np.zeros((1, 30), dtype=np.float32)

mock_model = MagicMock()
mock_model.return_value = mock_tensor

mock_scaler = MagicMock()
mock_scaler.transform.return_value = [[0.5]]

patches = [
    patch('torch.load', return_value={}),
    patch('joblib.load', return_value=mock_scaler),
    patch('torch.nn.Module.load_state_dict', return_value=None),
    patch('prometheus_client.registry.REGISTRY._names_to_collectors', {}),
]

for p in patches:
    p.start()

import main as main_module

main_module.model = mock_model
main_module.scaler = mock_scaler
main_module.feature_cols = [f'V{i}' for i in range(1, 29)] + ['Amount', 'Time']
main_module.best_threshold = 3.9326