from mxnet.gluon.model_zoo import vision as models
from mxnet.gluon.utils import download
from mxnet import image
import matplotlib.pyplot as plt
from mxnet import nd, gpu

GPU_ID = 1

net = models.resnet50_v2(pretrained=True, ctx=gpu(GPU_ID))
url = 'http://data.mxnet.io/models/imagenet/synset.txt'
fname = download(url)
with open(fname, 'r') as f:
    text_labels = [' '.join(l.split()[1:]) for l in f]
url='https://upload.wikimedia.org/wikipedia/commons/thumb/b/b5/Golden_Retriever_medium-to-light-coat.jpg/365px-Golden_Retriever_medium-to-light-coat.jpg'
fname = download(url)
x = image.imread(fname)
x = image.resize_short(x, 256)
x, _ = image.center_crop(x, (244, 244))
plt.imshow(x.asnumpy())
plt.show()
x = x.copyto(gpu(GPU_ID))
rgb_mean = nd.array([0.485, 0.456, 0.406], ctx=gpu(GPU_ID)).reshape((1,3,1,1))
rgb_std = nd.array([0.229,0.224,0.225], ctx=gpu(GPU_ID)).reshape((1,3,1,1))
def transform(data):
    data = data.transpose((2, 0, 1)).expand_dims(axis=0)
    return (data.astype('float32')/255 - rgb_mean) / rgb_std
prob = net(transform(x))
