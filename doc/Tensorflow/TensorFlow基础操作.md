# Tensorflow基础操作
### 创建tensor
#### 1. from Numpy,List
1. tf.convert_to_tensor(np.ones([2,3])) 
- 这是将一个2维3列的numpy列表转换为tensor类型
- 每个元素都赋值为1
- 通过numpy转换的每个元素都是默认的float64类型
- 通过shape赋值
2. tf.convert_to_tensor([1, 2.])
- 指定这是1维2列的列表
- 自定义元素的数据类型都是float32
- 元素未赋初值
- 通过shape赋值
3. tf.convert_to_tensor([[1], [2.]])
- 指定这是1维2列的matrix
- 通过data赋值
4. tf.convert_to_tensor(1, (1,2))
- 报错,因为tensor支持的数据类型只有四种,scalar,vector,matrix,tensor.
> 常用场景
##### 1. 初始化为0
- tf.zeros([]) 新建一个scalar,numpy=0.0,shape为()
- tf.zeros([1]) 新建一个vector,numpy=([0.], dtype=float32)
- tf.zeros([2,2]) 新建一个matrix, dbtype=float32
- tf.zeros([2,3,3]) 新建一个tensor,dbtype=float32
- 以上入参全是通过shape创建
- tf.zeros_like(a) 根据a的shape创建一个元素全0的矩阵
- tf.zeros(a.shape) 跟上面实现的功能是一致的
##### 2. 初始化为1
- tf.onse <==> tf.zeros
##### 3. 初始化为同一个给定值
- tf.fill([2,2], 3) 类似tf.ones, 给矩阵的每个元素初始化为指定值,这里是给3
##### 4. 随机初始化
- 均匀分布
- 正太分布:
tf.random.normal([2,2], mean=1, stddev=2) mean 均值 stddev 方差