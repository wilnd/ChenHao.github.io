# TensorFlow 
tensor 经过各种各样的运算,在网络中不断流动,最终得到我们要的结果,就叫TensorFlow
##### 容器
- list [] 可以放多种数据类型,但是容量有限
- np.arry 可以存大量数据
- tf.Tensor  可以存大量数据,并且具备tensor特性:求导
##### what is tensor
- scalar: 1.1 0维数据,标量
- vector: [1.1] dim = 1 , [1.1,2.2,...] 一维向量
- matrix: [[1.1,2.2], [3.3,4.4]] 行列式 二维
- tensor: rank > 2 维度 > 2叫tensor. 广义来说所有数据都是tensor,包括上面的0维,一维,二维数据
##### 基本数据类型
- int, float, double
>tf.constant(1) 创建一个整形常量
tf.constant(1.) 创建一个浮点型 floa32
tf.constant(2., dtype=tf.double) 创建一个float64类型

- bool
>tf.constant([True, False]) 创建一个bool型

- string
> tf.constant('hello world') 创建一个string类型

##### 常用属性
- with tf.device("cpu"):  以下操作在cpu设备上执行
- with tf.device('gpu'):  以下操作在gpu设备上执行
- 默认是在gpu设备上执行
- aa = a.gpu() 不管之前a在哪个设备上,后面都在gpu下操作
- bb = b.cpu() 不管之前b在哪个设备上,后面都在cpu下操作
- b.numpy() numpy跟tensor都是数据载体,需要支持相应转换, 这里是将b转成numpy为载体,当然必须在cpu上执行
- b.shape 返回一个矩阵
- b.ndim 返回数据的维度
- tf.rank(b) tensor类型,返回数据维度
- b.name 是第一个版本遗留下来爱的,其实就是它本身
- tf.is_tensor(b) 判断数据是否是tensor类型.返回true,false. (isinstance(a, tf.tensor)也能办到类似的事情)

#### 类型间转换
- numpy 整形默认是int64, 转换为tensor后也是64位
- tensor -> numpy 
>假如a是tensor
a.numpy
int(a)
float(a)

- tensor间数据类型转换时通过 tf.cast来进行的,只要不报错,都能转换如: 
>tf.cast(param, dtype=tf.float32)
>tf.cast(param, dtype=tf.double)
>tf.cast(param, dtype=tf.int32)



- bool <> int 
> ##### 整形转bool
b = tf.constant([0,1]) 定义一个整形
tf.cast(b, dtype=tf.bool)
##### bool转整形
b = tf.cast(b, dtype=tf.bool)
tf.cast(b, tf.int32)

- tf.variable
>tensor类型的数据被tf.Variable包装后组,就是可求导的.这是专门为神经网络设定的一个类型
b.trainable = true

