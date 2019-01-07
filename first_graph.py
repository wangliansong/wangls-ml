import tensorflow as tf
#tf.Variable生成的变量，每次迭代都会变化
#这个变量也就是我们要去计算的结果，所以说你要计算什么，就把什么定义成Variable
'''
Tensorflow程序可以通过tf.device函数来指定运行每一个操作的设备
这个设备可以是本地的CPU或者GPU，也可以是某一台远程的服务器。
TensorFlow会给每一台可用的设备一个名词，tf.device函数可以通过设备的名称，来指定运算的设备，
比如CPU在TensorFow中的名称为/cpu:0.
在默认情况下，即使机器有多个CPU，TensorFlow也不会区分他们，所有的CPU都使用/cpu:0作为名称。
而一台机器上不同的GPU的名称是不同的，第n个GPU在TensorFlow中的名称为/gpu:n
比如第一个GPU的名称为/gpu:0，第二个GPU的名称为/gpu:1,以此类推。
TensorFlow提供了一个快捷的方式，来查看运行每一个运算的设备。
在生成会话时，可以通过设置log_device_placement参数来打印运行每一个运算的设备。
除了可以看到最后的计算结果之外，还可以看到类似"add:/job:localhost/replica:0/task:0/cpu:0"这样的输出
这些输出显示了执行每一个运算的设备，比如加法操作add是通过CPU来运行的，因为它的设备名称中包含了/cpu:0.
在配置好GPU环境的TensorFlow中，如果操作没有明确的指定运行的设备，那么TensorFlow会优先选择GPU
'''
with tf.device('/cpu:0'):
    x = tf.Variable(3,name='x')
y = tf.Variable(4,name='y')
f = x*x*y +y +2
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
sess.run(x.initializer)
sess.run(y.initializer)
result=sess.run(f)
print(result)
sess.close()
