import paddle.fluid as fluid
import numpy as np
import paddle.fluid.layers as L
def gen_data():
    return {
        "x": np.random.randint(1, 5, size=[8, 10]).astype('float32'),
        "y": np.random.randint(1, 5, size=[10]).astype('float32'),
    }
x = fluid.layers.data(name="x", shape=[8,10], dtype='float32')
y = fluid.layers.data(name="y", shape=[10], dtype='float32')
mm = L.sqrt(L.reduce_sum(L.elementwise_mul(x,x), dim=0))
kk = L.ones_like(y)
z = fluid.layers.elementwise_div(x, mm, axis=1)
# z = x / y
place = fluid.CPUPlace()
exe = fluid.Executor(place)
z_value = exe.run(feed=gen_data(),
                    fetch_list=[z.name])
print(z_value) #