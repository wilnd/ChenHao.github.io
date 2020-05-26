# Tensorflow基础操作
### 1.创建tensor
#### 1.1常用创建方式
>#####  tf.convert_to_tensor(np.ones([2,3])) 
- 这是将一个2维3列的numpy列表转换为tensor类型
- 每个元素都赋值为1
- 通过numpy转换的每个元素都是默认的float64类型
- 通过shape赋值
##### tf.convert_to_tensor([1, 2.])
- 指定这是1维2列的列表
- 自定义元素的数据类型都是float32
- 元素未赋初值
- 通过shape赋值
##### tf.convert_to_tensor([[1], [2.]])
- 指定这是1维2列的matrix
- 通过data赋值
##### tf.convert_to_tensor(1, (1,2))
- 报错,因为tensor支持的数据类型只有四种,scalar,vector,matrix,tensor.

#### 1.2常用场景:
>##### 初始化为0
- tf.zeros([]) 新建一个scalar,numpy=0.0,shape为()
- tf.zeros([1]) 新建一个vector,numpy=([0.], dtype=float32)
- tf.zeros([2,2]) 新建一个matrix, dbtype=float32
- tf.zeros([2,3,3]) 新建一个tensor,dbtype=float32
- 以上入参全是通过shape创建
- tf.zeros_like(a) 根据a的shape创建一个元素全0的矩阵
- tf.zeros(a.shape) 跟上面实现的功能是一致的
##### 初始化为1
- tf.onse <==> tf.zeros
##### 初始化为同一个给定值
- tf.fill([2,2], 3) 类似tf.ones, 给矩阵的每个元素初始化为指定值,这里是给3
##### 4. 随机初始化
- 均匀分布
tf.random.uniform([2,2],minval=0,maxval=1) minval 最小值,maxval 最大值
- 正太分布:
tf.random.normal([2,2], mean=1, stddev=2) mean 均值 stddev 方差
tf.random.truncated_normal([2,2], mean=0,stddev=1) 截断后的正态分布, 截去一段区域后重新取样
Gradient Vanish 梯度接近于0,叫做梯度消失,进行数据训练非常困难.所以这个时候需要截去一部分

#### 1.3常见函数
>- tf.gather(a, idx) 将a的数据按照idx的顺序打散
- tf.constant(1) 创建一个0维的标量
- tf.constant([1]) 创建一个一维的向量. 2,3 .. 维类似. 须保证数据结构
- tf.range(4) 创建一个0-4的向量
- tf.one_hot(y, detpth=10) 将y转换为one-hot编码
- tf.keras.losses.mse(y,out) 将y与out求空间距离(MSE)
- tf.reduce_mean(loss) loss一般是一个一维数据,这个方法是将loss数据求平均值,所以结果是标量
- net = layers.Dense(10) 设置一个网络,将维度转换为10
- net.build((4,8)) 设置网络的输入维度为4行8列的matrix,net的能力是将输入转换为4行10列的matrix
- net(x).shape 实际转换过程
- net.kernel 维度为2的
- net.bias 维度为1的
- layers.Conv2D(16,kernel_size=3) 卷积函数,将3变为16

### 2.索引与切片
#### 2.1索引
##### a=tf.random.normal([4,28,28,3])
- a[0][0][0][2] 给定各个维度的索引 
- a[1] = a[0,:,:,:]-> [28,28,3]
- a[1][2] -> [28,3]
- a[1,2,3,2] -> []
#### 2.2 切片
- 一个逗号代表一个维度
- : 每个维度切片用冒号分割
>- a[-1:] 从倒数第一个位置的数值到第一个位置的数据
- a[:2] 从最开始到第2个数字,左闭右开
- a[:-1]从最开始到最后一个数字,左闭右开
- a[A:B:-1] 从A到B开始逆着采样

- :: 两个冒号代表倍数
>- a[:,::2,::2,:] 对于所有的行,所有的列,隔着采样

- ... 代表多个:
>a.tf.random.normal([2,4,28,28,3])
- a[0,...,2] -> [4, 28, 28]

#### 2.3 常用函数
a = [4,35,8] -> class,student,subject
#####  tf.gather 按维度收集数据
- tf.gather(a,axis=0, indices=[2,3]) 抽取第2,3个班级
- tf.gather(a,axis=0,indices=[2,1,4,0]) 按照指定顺序抽取2,1,4,0这些班级的学生的成绩
- tf.gather(a,axis=1,incices=[2,3,7,9,16]) 抽取所有班级的2,3,7,9,16号学生的所有成绩
- tf.gather(a,axis=2,incices=[2,3,7]) 抽取第2,3,7门功课的成绩
- tf.gather()
> - axis:维度
- incices: 收集的数据

#####  tf.gather_nd 按矩阵收集数据
a = tf.random.normal([4,35,8])
- tf.gather_nd(a,[0]) -> [35,8]
- tf.gather_nd(a,[0,1]) -> [8]
- tf.gather_nd(a,[0,1,2]) -> []
- tf.gather_nd(a,[[0,0],[1,1]]) -> [2,8] 两个8维的vector组成
- tf.gather_nd(a,[[[0,0,0],[1,1,1],[2,2,2]]]) -> [3] 三个0维vector组成
> 
1. 一个中括号代表取一个样本
2. 中括号里面的数据代表每个样本的维度
3. 多个逗号分隔中括号最终得到指定数 
4. 多个中括号嵌套的表达式最终得到更低维的数据
5. 建议采用两层嵌套的中括号,不论精确到哪个维度

#####  tf.boolean_mask 过滤数据
a = [4,28,28,3]
- tf.boolean_mask(a,mask=[True,True,False], axis=3) ->[4,28,28,2]
- tf.boolean_mask(a, mask=[True,True,False,False]) -> [2,28,28,3]
a = tf.ones(2,3,4)
- tf.boolean_mask(a,mask=[[True,False,False],[False,True,True]) -> [3,4]
>mask2行3列,对应a的2行3列. 总共采到3个样本,每个样本的维度是4

### 3. 维度变换
- shape 维度
- axix 轴
- 轴的理解,可以带入实际有物理意义的东西 [b,28,28,1]->[batch,row(height),column(width),channel]
##### 3.1 reshape 维度变换的操作
> 1. [b,28,28] 
2.  [b, 28*28]
3. [b, 2, 14*28]
4. [b, 28,28,1]
view的转换需要注意:
1. 不要修改原来的数据
2. 把原来的数据利用完
3. reshape有很多种可能性,但是最好进行的变换具备相应的物理意义.
4. reshape会丢失掉信息,如果想将shape转回去,需要将丢失的信息加进去
实例:
1. a = tf.random.normal([4,28,28,3])
2. a.shape,a.ndim -> 输出形状,维度信息
3. tf.reshape(a,[4,784,3]) -> 将a的形状进行变更 [b,pixel,c] 将行列信息抹掉添加一个像素维度
4. tf.reshape(a,[4,-1,3]) -> 如果不想计算784,可以用-1去代替, 会自动计算得到-1实际值.一个表达式只能有一个-1
5. tf.reshape(a,[4,784*3]) -> [4,2352] 将行列,像素的数据抹掉,转换为数据点的概念.同样可以写成[4,-1]

##### 3.2 tf.transpose 默认矩阵的转置,沿着主对角线进行交换
> a = tf.random.normal([4,3,2,1])
1. tf.transpose(a) -> [1,2,3,4]
2. tf.transpose(a, perm=[0,1,3,2]) ->[4,3,1,2]

##### 3.3 expand dim 增加维度
> a = tf.random.normal([4,35,8])
1. tf.expand_dims(a,axis=0) -> [1,4,35,8]  0号位置增加
2. tf.expand_dims(a,axis=3) -> [4,35,8,1] 3号位置增加
3. tf.expand_dims(a,axis=-1) -> [4,35,8,1] 在最后一个位置增加
4. tf.expand_dims(a,axis=-4) -> [1,4,35,8] 在倒数第4个位置增加
5. 正数新增维度占据当前位置,负数新增维度向右偏

##### 3.4 Squeeze dim 减少维度
>1. tf.squeeze(tf.zeros([1,2,1,1,3])) -> [2,3] 默认去掉所有shape=1的
tf.zeros([1,2,1,3])
2. tf.squeeze(a,axis=0) -> [2,1,3] 
3. tf.squeeze(a,axis=-2) -> [1,2,3]

### 4. Broadcasting 
- 默认使用Broadcasting能力
- 调用函数tf.broadcast_to 显示使用Broadcasting能力
##### 4.1 AB维度不一致时进行运算会自动BroadCasting
>- 小维度按右对齐,没有的地方补1
- 把1扩展成相对应的数,看上去扩展,实际没有扩展
- 进行运算
- 实际意义是A的各个高纬度对应的所有低维度都跟B进行相加

##### 4.2 Broadcasting = tf.expand_dims + tf.tile
> 但是使用吗偶人的BroadCast会更好
1. 所占据的内存空间会更少
2. 写起来更简单

##### 4.3 使用Broadcasting的条件
>是否能够自动BroadCasting 取决于向右对齐后低维度的数据维度是否相同

### 5.运算
#### 5.1element-wise 对应元素的加减乘除
- +-*/(加减乘除)
> b=tf.fill([2,2],2.);a=tf.ones([2,2])
= a+b -> [[3,3],[3,3]]
- a-b -> [[-1,-1],[-1,-1]]
- a*b -> [[2.,2.],[2.,2.]]
- a/b -> [[0.5,0.5],[0.5,0.5]]

- //,%(取整,取余)
>- b//a -> [[2.,2.],[2.,2.]]
- b%a -> [[0.,0.],[0.,0.]]

- exp,log(e为底的指数,e为底的对数)
>a=tf.ones([2,2])
- tf.math.log(a) -> [[0.,0.],[0.,0.]]
- tf.exp(a) -> [[2.7182817,2.7182817],[2.7182817,2.7182817]]
- tf.math.log是以自然指数为底的

- pow,sqrt,**(指数,开方,指数)
> b=tf.fill([2,2],2.)
- tf.pow(b,3) -> [[8.,8.],[8.,8.]]
- b**3 -> [[8.,8.],[8.,8.]]
- tf.sqrt(b) -> [[1.4142135,1.4142135],[1.4142135,1.4142135]]

#### 5.2. matrix-wise 两个矩阵间的计算
- @ matmul(矩阵相乘)
##### b=tf.fill([2,2],2.);a=tf.ones([2,2])
>- a@b -> [[4,4],[4,4]]
- tf.matmul(a,b) -> [[4.,4.],[4.,4.]]
- a=[b,2,3] b=[b,3,5]时a@b = [b,2,5]
- a=[4,2,3] b=[3,5]时 先bb = tf.broadcast_to(b,[4,3,5])后a@b

#### 5.3 dim-wise对某一个维度进行操作
- y = w*x + b
- y = w@x + b 其实这种矩阵相乘后的y展开后,是多个y = w*x + b 的式子的和
- y = tf.nn.relu(y) 这个是非线性因子, 加上这个非线性因子后,会将所有负数干掉,类似乘上一个掩码

### 6.练习
```
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 自动在网上加载一个数据集
# x: [60k, 28, 28]
# y: [60k]
(x, y), _ = datasets.mnist.load_data()

# X:[0~255] => [0~1.]
x = tf.convert_to_tensor(x, dtype=tf.float32) / 255.
y = tf.convert_to_tensor(y, dtype=tf.int32)

print(x.shape, y.shape, x.dtype, y.dtype)
print(tf.reduce_min(x), tf.reduce_max(x))
print(tf.reduce_min(y), tf.reduce_max(y))

train_db = tf.data.Dataset.from_tensor_slices((x, y)).batch(128)
train_iter = iter(train_db)
sample = next(train_iter)
print('batch:', sample[0].shape, sample[1].shape)

# 设置初值
# [b,784] => [b, 512] => [b , 128] => [b, 10]
# [dim_in, dim_out], [dim_out]
w1 = tf.Variable(tf.random.truncated_normal([784, 512], stddev=0.1))  # 随机初值是正态分布
b1 = tf.Variable(tf.zeros([512]))  # 随机初值是0
w2 = tf.Variable(tf.random.truncated_normal([512, 128], stddev=0.1))
b2 = tf.Variable(tf.zeros([128]))
w3 = tf.Variable(tf.random.truncated_normal([128, 10], stddev=0.1))
b3 = tf.Variable(tf.zeros([10]))

lr = 1e-3
for epoch in range(10):  # 整个数据集迭代10次
    for step, (x, y) in enumerate(train_db):  # for every batch
        # x: [128, 28, 28]
        # y: [128]
        # [b, 28, 28] => [b, 28*28]
        x = tf.reshape(x, [-1, 28 * 28])

        # 3. 利用TensorFlow梯度计算 自动求导
        with tf.GradientTape() as tape:  # 默认只会跟踪tf.Variable类型的数据,所以求梯度的时候需要转换成这个类型
            # 1. 前向计算
            # x: [b, 28*28]
            # h1 = x@w1 + b1
            # [b,784]@[784,512]+[512] => [b,256] + [256] => [b, 256] + [b, 256]
            h1 = x @ w1 + tf.broadcast_to(b1, [x.shape[0], 512])  # 这个broadcast是多此一举,主要是其实不写默认也是这样执行
            h1 = tf.nn.relu(h1)
            # [b, 256] => [b, 128]
            h2 = h1 @ w2 + b2
            h2 = tf.nn.relu(h2)
            # [b, 128] => [b, 10]
            out = h2 @ w3 + b3

            # 2. compute loss 计算误差
            # out: [b, 10]
            # y: [b] = > [b, 10]
            y_one_hot = tf.one_hot(y, depth=10)

            # mse = mean(sum(y-out)^2)
            # [b, 10]
            loss = tf.square(y_one_hot - out)
            # mean: scalar 求均值
            loss = tf.reduce_mean(loss)

        # compute gradients 求梯度
        grads = tape.gradient(loss, [w1, b1, w2, b2, w3, b3])

        # 4. 找最合适值得过程
        # w1 = w1 - lr * w1_grad
        # w1 = w1 - lr * grads[0] # 相减后会将tf.Variable类型返回一个tensor类型,需要原地更新 w1.assign_sub
        # b1 = b1 - lr * grads[1]
        # w2 = w2 - lr * grads[2]
        # b2 = b2 - lr * grads[3]
        # w3 = w3 - lr * grads[4]
        # b3 = b3 - lr * grads[5]
        w1.assign_sub(lr * grads[0])
        b1.assign_sub(lr * grads[1])
        w2.assign_sub(lr * grads[2])
        b2.assign_sub(lr * grads[3])
        w3.assign_sub(lr * grads[4])
        b3.assign_sub(lr * grads[5])

        if step % 100 == 0:
            print(epoch, step, 'loss:', float(loss))

```
