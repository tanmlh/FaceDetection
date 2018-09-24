import cv2
import mxnet as mx
import mxnet.ndarray as nd

def get_ctx():
    """ get the current context """
    return mx.gpu()

def show_img(img_np):
    cv2.imshow('a', img_np)
    k = cv2.waitKey(0)
    if k == ord('q'):
        cv2.destroyAllWindows()

def chg_ctx(arg_params, aux_params, ctx):
    """change the context of dict-like parameters"""
    new_args = dict()
    new_auxs = dict()
    for key, value in arg_params.items():
        new_args[key] = value.as_in_context(ctx)
    for key, value in aux_params.items():
        new_auxs[key] = value.as_in_context(ctx)
    return new_args, new_auxs

def np2nd(img_np, ctx=get_ctx()):
    img_nd = nd.array(img_np, ctx=ctx)
    img_nd = nd.swapaxes(img_nd, 1, 2)
    img_nd = nd.swapaxes(img_nd, 0, 1)
    img_nd = nd.expand_dims(img_nd, 0)
    return img_nd

def resize_pad(self, img_np, des_size):
    ratio_src = img_np.shape[0] / img_np.shape[1]
    ratio_des = des_size[0] / des_size[1]
    if ratio_src > ratio_des:
        scale = des_size[0] / img_np.shape[0]
    else:
        scale = des_size[1] / img_np.shape[1]
    img_np = cv2.resize(img_np, None, None, fx=scale, fy=scale,\
                        interpolation=cv2.INTER_LINEAR)
    if ratio_src > ratio_des:
        delta = des_size[1]-img_np.shape[1]
        pad = (0, 0, delta//2, delta-delta//2)
    else:
        delta = des_size[0]-img_np.shape[0]
        pad = (delta//2, delta-delta//2, 0, 0)
    img_np = cv2.copyMakeBorder(img_np, pad[0], pad[1],\
                                pad[2], pad[3],\
                                cv2.BORDER_CONSTANT, value=(0,0,0))
    return img_np
