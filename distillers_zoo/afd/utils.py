# Attention-based Feature-level Distillation
# Original Source : https://github.com/HobbitLong/RepDistiller

import numpy as np

LAYER = {'resnet20': np.arange(1, (20 - 2) // 2 + 1),  # 9
         'resnet56': np.arange(1, (56 - 2) // 2 + 1),  # 27
         'resnet110': np.arange(2, (110 - 2) // 2 + 1, 2),  # 27
         'wrn40x2': np.arange(1, (40 - 4) // 2 + 1),  # 18
         'wrn28x2': np.arange(1, (28 - 4) // 2 + 1),  # 12
         'wrn16x2': np.arange(1, (16 - 4) // 2 + 1),  # 6
         'resnet34': np.arange(1, (34 - 2) // 2 + 1),  # 16
         'resnet18': np.arange(1, (18 - 2) // 2 + 1),  # 8
         'resnet34im': np.arange(1, (34 - 2) // 2 + 1),  # 16
         'resnet18im': np.arange(1, (18 - 2) // 2 + 1),  # 8
         }


def unique_shape(s_shapes):
    n_s = []
    unique_shapes = []
    n = -1
    for s_shape in s_shapes:
        if s_shape not in unique_shapes:
            unique_shapes.append(s_shape)
            n += 1
        n_s.append(n)
    return n_s, unique_shapes
