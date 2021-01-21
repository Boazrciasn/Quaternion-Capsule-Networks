import torch
import torch.nn as nn
import torch.nn.functional as F
import math

eps = 1e-10

# Layer template using quaternion ops, has quatembedder and left2right for quaternion operations.
# quadembedder: Matrix embedding for quaternion q1 to apply quaternion product from left (q1 . q').
# left2right: matrix embedding for convertin quaternion product from left matrix to from right matrix (q' . q1).
class QuaternionLayer(nn.Module):
    def __init__(self):
        super(QuaternionLayer, self).__init__()
        self.register_buffer("quatEmbedder", torch.stack([torch.eye(4),
                                                          torch.tensor([[0, -1, 0, 0],
                                                                        [1, 0, 0, 0],
                                                                        [0, 0, 0, -1],
                                                                        [0, 0, 1, 0]], dtype=torch.float),

                                                          torch.tensor([[0, 0, -1, 0],
                                                                        [0, 0, 0, 1],
                                                                        [1, 0, 0, 0],
                                                                        [0, -1, 0, 0]], dtype=torch.float),

                                                          torch.tensor([[0, 0, 0, -1],
                                                                        [0, 0, -1, 0],
                                                                        [0, 1, 0, 0],
                                                                        [1, 0, 0, 0]], dtype=torch.float)]).unsqueeze(0).unsqueeze(1))
        # convert quaternion multiplication from left to multiplication from right.
        self.register_buffer("left2right", torch.tensor([[1, 1, 1, 1],
                                                         [1, 1, -1, -1],
                                                         [1, -1, 1, -1],
                                                         [1, -1, -1, 1]], dtype=torch.float)[(None,) * 5])


if __name__ == "__main__":
    dummy_x = torch.rand([2, 4, 4, 16, 4, 1])
    dummy_a = torch.rand([2, 4, 4, 16, 1])
    test_layer = STRoutedQCLayer(inCaps=16, outCaps=5, quat_dims=3)
    out = test_layer(dummy_x, dummy_a)
