#!/usr/bin/env python
# Created by "Thieu" at 14:15, 25/05/2023 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from metacluster import get_dataset

np.random.seed(41)


def test_Data_class():
    data = get_dataset("circles")
    data.X, scaler = data.scale(data.X, method="MinMaxScaler", feature_range=(0, 1))

    assert 0 <= data.X[np.random.randint(0, 3)][0] <= 1
