# IScope模块

用来管理tensorrt网络中权重的命名空间。其中公有成员函数有：

## IScope()

函数定义：
```
IScope()
```
构造函数：初始化网络权重的命名空间，初始化为空。

## IScope()

函数定义：

```
IScope(const std::string& name)
```
参数：
* name：用来初始化网络权重的名字

构造函数：初始化网络权重的根结点命名空间，初始化为name

## getScopeName()

函数定义：

```
std::string getScopeName() const
```
返回当前scope的名字

## getOpName()

函数定义：

```
std::string getOpName() const
```
返回当前scope的op名字，op名字为scope_name + ":0"

## operator=()

函数定义：

```
IScope& operator=(const IScope& other)
```
参数：
* other：用来赋值的scope

IScope类之间赋值操作

## subIScope()

函数定义：

```
IScope subIScope(const std::string& child_scope_name)
```
参数：
* child_scope_name：网络权重子节点的名字

在上一级网络权重的名字上加上子节点的名字，返回子节点的命名空间IScope






