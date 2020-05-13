# Python
我之前有在菜鸟教程通读一遍python基础知识,这里不会再系统的去学习python的用法,而是每次遇到不懂得语法,属性或函数会记录下来.当然,如果某些知识点收集的时候发现篇幅过长,我还是会将他们单独放到一个专门的文件里面,不占用太多的篇幅
##### 1. data.shape 
- 查看矩阵或者数组的维数
##### 2. with...as 
```
class test():
    def __init__(self):
        self.text = "hello"
    def __enter__(self):
        self.text += " world"
        return self              #这句必须要有，不然with ... as 时，as后面的变量没法被赋值
    def __exit__(self, arg1, arg2, arg3):     #一共四个参数，后面四个参数是好像是关于异常信息的，没研究过，先这样写着
        self.text += "!"
    def Print(self):
        print self.text

try:
    with test() as f:      #在with ... as的作用域内，进入会执行test()的__enter__()函数，出作用域执行__exit__()函数
        f.Print()
        raise StopIteration

except StopIteration:
    f.Print()
--------------------------------------------------
运行结果:
hello world
hello world!
```
- 在with...as的作用域内,进入会先执行test()的__enter__()函数,出去会执行__exit__()
- 如果在作用域内发生异常,出去的时候,__enter__()函数仍然会执行
- f的作用域并不局限于with ... as内
- test类中函数的执行顺序是 __init__() -->__enter__() --> f.Print() --> __exit__