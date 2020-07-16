import torch
import numpy as np
import paddle
import paddle.fluid as F
import paddle.fluid.dygraph as FD
import paddle.fluid.layers as L

x = L.data(name='x', shape=[8, 10], dtype='float32')
hidden1 = norm(x, dim=0)
place = F.CPUPlace()
exe = F.Executor(place)
exe.run(F.default_startup_program())
np_x = np.random.random(size=(8, 10)).astype('float32')
kk, output = exe.run(feed={"x": np_x}, fetch_list = [x, hidden1])
print(output)
tokk = torch.from_numpy(kk)
toout = torch.nn.functional.normalize(tokk, dim=0)
print(toout)

