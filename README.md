文件夹下共20个文件，分别是Client1-20的隐私训练数据。

读取方式如下，以读取Client1.pkl为例：

```python
with open("XXX\\Client1.pkl",'rb') as f:
    train_dataset_client_1 = dill.load(f)
```

（把XXX改为你的下载路径哦）
需要提前装好dill（pip install dill）并在开头import dill。
注意，这个读取出来的是dataset，可以输出其长度，即len(train_dataset_client_1)，另外其中的每一个数据的格式为<一个1*28*28的图片，一个类别标签>。


另外，用于评估global model的测试数据这里没有提供，请自己使用代码下载。

```python
test_dataset = torchvision.datasets.MNIST(
        './data', train=False, download=True, transform=transform
    )
```

使用上面的代码会把测试数据集下载到代码所在目录中的data文件夹下。