from mxnet import nd, gluon, init, autograd, gpu
from mxnet.gluon import nn
from mxnet.gluon.data.vision import datasets, transforms
import matplotlib.pyplot as plt
from time import time
# dfsdf

mnist_train = datasets.FashionMNIST(train=True)

transformer = transforms.Compose([transforms.ToTensor(),
                                 transforms.Normalize(0.13, 0.31)])
mnist_train = mnist_train.transform_first(transformer)
batch_size = 256
train_data = gluon.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=4)

mnist_valid = gluon.data.vision.FashionMNIST(train=False)
valid_data = gluon.data.DataLoader(mnist_valid.transform_first(transformer), batch_size=batch_size,
                                   num_workers=4)
net = nn.Sequential()
net.add(nn.Conv2D(channels=6, kernel_size=5, activation='relu'),
        nn.MaxPool2D(pool_size=2, strides=2),
        nn.Conv2D(channels=16, kernel_size=3, activation='relu'),
        nn.MaxPool2D(pool_size=2, strides=2),
        nn.Flatten(),
        nn.Dense(120, activation='relu'),
        nn.Dense(84, activation='relu'),
        nn.Dense(10))
net.initialize(init=init.Xavier())

softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate':0.1})

def acc(output, label):
    return (output.argmax(axis=1) == label.astype('float32')).mean().asscalar()

for epoch in range(10):
    train_loss, train_acc, valid_acc = 0., 0., 0.
    tic = time()
    for data, label in train_data:
        with autograd.record():
            output = net(data)
            loss = softmax_cross_entropy(output, label)
        loss.backward()
        trainer.step(batch_size)
        train_loss += loss.mean().asscalar()
        train_acc += acc(output, label)
    for data, label in valid_data:
        valid_acc += acc(net(data), label)
    print("Epoch %d: Loss: %.3f, Train acc %.3f, Test acc %.3f, Time %.1f sec" %
          (epoch,
           train_loss/len(train_data),
           train_acc/len(train_data),
           valid_acc/len(valid_data),
           time()-tic))
net.save_parameters('../../Model/Face_Detection/test_mxnet.params');


