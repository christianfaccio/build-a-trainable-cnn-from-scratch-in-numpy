"""
Build a Trainable CNN from Scratch in NumPy

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - argmax_rows
def argmax_rows(matrix):
    # TODO: return the index of the largest element in each row of a 2D array
    return np.argmax(matrix, axis=1)

# Step 2 - row_max
import numpy as np

def row_max(matrix):
    # TODO: return the maximum value of each row of `matrix` with keepdims True for broadcasting.
    return np.max(matrix, axis=1, keepdims=True)

# Step 3 - row_sum
import numpy as np

def row_sum(matrix):
    """Return per-row sums of a 2D array with shape (N, 1)."""
    # TODO: return the sum along axis 1 keeping the reduced dimension
    return np.sum(matrix, axis=1, keepdims=True)

# Step 4 - exp_shifted
import numpy as np

def row_max(matrix):
    # TODO: return the maximum value of each row of `matrix` with keepdims True for broadcasting.
    return np.max(matrix, axis=1, keepdims=True)

def exp_shifted(logits):
    """Subtract per-row max from logits and exponentiate elementwise."""
    # TODO: shift each row of logits by its max and return elementwise exp
    return np.exp(logits - row_max(logits))

# Step 5 - stable_softmax
def row_sum(matrix):
    """Return per-row sums of a 2D array with shape (N, 1)."""
    # TODO: return the sum along axis 1 keeping the reduced dimension
    return np.sum(matrix, axis=1, keepdims=True)

def exp_shifted(logits):
    """Subtract per-row max from logits and exponentiate elementwise."""
    # TODO: shift each row of logits by its max and return elementwise exp
    return np.exp(logits - row_max(logits))

def stable_softmax(logits):
    # TODO: Compute a numerically stable softmax row-wise over (N, C) logits.
    return exp_shifted(logits) / row_sum(exp_shifted(logits))

# Step 6 - one_hot
def one_hot(labels, num_classes):
    # TODO: convert integer labels into a (N, num_classes) one-hot float matrix
    out = np.zeros((labels.shape[0], num_classes), dtype=np.float64)
    for i in range(labels.shape[0]):
        for j in range(num_classes):
            if labels[i] == j:
                out[i][j] = 1.0
    return out

# Step 7 - gather_true_class_probs
def gather_true_class_probs(probs, labels):
    # TODO: return probs[i, labels[i]] for every row i as a 1D length-N array.
    out = np.zeros(probs.shape[0], dtype=np.float64)
    for i in range(probs.shape[0]):
        for j in range(probs.shape[1]):
            if labels[i] == j:
                out[i] = probs[i][j]
    return out

# Step 8 - cross_entropy_loss
import numpy as np

def gather_true_class_probs(probs, labels):
    # TODO: return probs[i, labels[i]] for every row i as a 1D length-N array.
    out = np.zeros(probs.shape[0], dtype=np.float64)
    for i in range(probs.shape[0]):
        for j in range(probs.shape[1]):
            if labels[i] == j:
                out[i] = probs[i][j]
    return out

def cross_entropy_loss(probs, labels, eps=1e-12):
    # TODO: return the mean negative log-likelihood of the true-class probabilities
    return -np.sum(np.log(np.clip(gather_true_class_probs(probs, labels), eps, None))) / probs.shape[0]

# Step 9 - accuracy
def argmax_rows(matrix):
    # TODO: return the index of the largest element in each row of a 2D array
    return np.argmax(matrix, axis=1)

def accuracy(logits_or_probs, labels):
    # TODO: return the fraction of rows whose argmax matches the integer label.
    predicted = argmax_rows(logits_or_probs)
    return np.sum(predicted == labels) / logits_or_probs.shape[0]

# Step 10 - he_std
def he_std(fan_in):
    # TODO: return the He initialization standard deviation sqrt(2 / fan_in).
    return np.sqrt(2/fan_in)

# Step 11 - he_init
def he_std(fan_in):
    # TODO: return the He initialization standard deviation sqrt(2 / fan_in).
    return np.sqrt(2/fan_in)

def he_init(shape, fan_in, seed):
    # TODO: sample a weight tensor from a normal distribution scaled by He std using the seed.
    np.random.seed(seed)
    std = he_std(fan_in)
    return np.random.normal(0, std, shape)

# Step 12 - init_zero_bias
import numpy as np

def init_zero_bias(length):
    # TODO: return a 1D float array of zeros with the given length.
    return np.zeros(length, dtype=np.float64)

# Step 13 - pad_2d
def pad_2d(images, pad):
    # TODO: zero-pad the spatial (H, W) dims of a 4D (N, C, H, W) tensor by `pad` on each side.
    if pad == 0:
        return images 
    out = np.zeros((images.shape[0], images.shape[1], images.shape[2]+(2*pad), images.shape[3]+(2*pad)), dtype=images.dtype)
    
    out[:,:,pad:pad+images.shape[2],pad:pad+images.shape[3]] = images
    return out

# Step 14 - output_spatial_size
def output_spatial_size(input_size, kernel, stride, padding):
    # TODO: return the conv/pool output spatial dimension from input_size, kernel, stride, padding
    return int(np.floor((input_size + 2 * padding - kernel) / stride) + 1)

# Step 15 - im2col
def pad_2d(images, pad):
    # TODO: zero-pad the spatial (H, W) dims of a 4D (N, C, H, W) tensor by `pad` on each side.
    if pad == 0:
        return images 
    out = np.zeros((images.shape[0], images.shape[1], images.shape[2]+(2*pad), images.shape[3]+(2*pad)), dtype=images.dtype)
    
    out[:,:,pad:pad+images.shape[2],pad:pad+images.shape[3]] = images
    return out

def output_spatial_size(input_size, kernel, stride, padding):
    # TODO: return the conv/pool output spatial dimension from input_size, kernel, stride, padding
    return int(np.floor((input_size + 2 * padding - kernel) / stride) + 1)

def im2col(images, kernel_h, kernel_w, stride, padding):
    # TODO: Unroll overlapping patches of a 4D image tensor into a 2D column matrix.
    N,C,H,W = images.shape
    x = pad_2d(images, padding)
    out_h = output_spatial_size(H, kernel_h, stride, padding)
    out_w = output_spatial_size(W, kernel_w, stride, padding)

    out = np.empty((N*out_h*out_w, C*kernel_h*kernel_w), dtype=images.dtype)
    counter = 0
    for i in range(N):
        for j in range(out_h):
            for k in range(out_w):
                hs,ws = j*stride, k*stride 
                out[counter] = x[i,:,hs:hs+kernel_h,ws:ws+kernel_w].reshape(-1)
                counter += 1
    
    return out

# Step 16 - col2im
def output_spatial_size(input_size, kernel, stride, padding):
    # TODO: return the conv/pool output spatial dimension from input_size, kernel, stride, padding
    return int(np.floor((input_size + 2 * padding - kernel) / stride) + 1)

def col2im(cols, input_shape, kernel_h, kernel_w, stride, padding):
    # TODO: re-roll a (N*out_h*out_w, C*kh*kw) column matrix back into a (N, C, H, W) tensor
    N,C,H,W = input_shape
    out_w = output_spatial_size(W, kernel_w, stride, padding)
    out_h = output_spatial_size(H, kernel_h, stride, padding)
    acc = np.zeros((N,C,H+2*padding,W+2*padding), dtype=cols.dtype)
    blocks = cols.reshape(N, out_h, out_w, C, kernel_h, kernel_w)
    for i in range(kernel_h):
        for j in range(kernel_w):
            acc[:,:,i:i+stride*out_h:stride,j:j+stride*out_w:stride] += blocks[:, :, :, :, i, j].transpose(0, 3, 1, 2)
    
    return acc[:,:,padding:H+padding,padding:W+padding]

# Step 17 - conv2d_forward
def output_spatial_size(input_size, kernel, stride, padding):
    # TODO: return the conv/pool output spatial dimension from input_size, kernel, stride, padding
    return int(np.floor((input_size + 2 * padding - kernel) / stride) + 1)

def im2col(images, kernel_h, kernel_w, stride, padding):
    # TODO: Unroll overlapping patches of a 4D image tensor into a 2D column matrix.
    N,C,H,W = images.shape
    x = pad_2d(images, padding)
    out_h = output_spatial_size(H, kernel_h, stride, padding)
    out_w = output_spatial_size(W, kernel_w, stride, padding)

    out = np.empty((N*out_h*out_w, C*kernel_h*kernel_w), dtype=images.dtype)
    counter = 0
    for i in range(N):
        for j in range(out_h):
            for k in range(out_w):
                hs,ws = j*stride, k*stride 
                out[counter] = x[i,:,hs:hs+kernel_h,ws:ws+kernel_w].reshape(-1)
                counter += 1
    
    return out

def conv2d_forward(x, weights, bias, stride, padding):
    # TODO: convolve x with weights using im2col, add bias, return output and a backprop cache.
    C_out, C_in, kh, kw = weights.shape # dim (C_out,C_in,kh,kw)
    weights = weights.reshape(C_out,C_in*kh*kw) # dim (C_in*kh*kw,C_out)
    N,C,H,W = x.shape
    out_h = output_spatial_size(H,kh,stride,padding)
    out_w = output_spatial_size(W,kw,stride,padding)
    col = im2col(x, kh, kw, stride, padding)    # dim (N*out_h*out_w,C_in*kh*kw)
    out = col @ weights.T + bias 
    return out.reshape(N,out_h,out_w,C_out).transpose(0,3,1,2), {
        'x_shape': x.shape,
        'weights': weights,
        'cols': col,
        'stride': stride,
        'padding': padding,
        'kernel_h': kh,
        'kernel_w': kw 
    }

# Step 18 - conv2d_grad_input
def col2im(cols, input_shape, kernel_h, kernel_w, stride, padding):
    # TODO: re-roll a (N*out_h*out_w, C*kh*kw) column matrix back into a (N, C, H, W) tensor
    N,C,H,W = input_shape
    out_w = output_spatial_size(W, kernel_w, stride, padding)
    out_h = output_spatial_size(H, kernel_h, stride, padding)
    acc = np.zeros((N,C,H+2*padding,W+2*padding), dtype=cols.dtype)
    blocks = cols.reshape(N, out_h, out_w, C, kernel_h, kernel_w)
    for i in range(kernel_h):
        for j in range(kernel_w):
            acc[:,:,i:i+stride*out_h:stride,j:j+stride*out_w:stride] += blocks[:, :, :, :, i, j].transpose(0, 3, 1, 2)
    
    return acc[:,:,padding:H+padding,padding:W+padding]

def conv2d_grad_input(d_out, cache):
    # TODO: backprop d_out through the conv input using col2im
    # d_out dims: (N,C_out,out_h,out_w)
    N,C_out,out_h,out_w = d_out.shape
    dY = d_out.transpose(0,2,3,1)
    dY = dY.reshape(-1,C_out)   # shape (N*out_h*out_w)
    # weights has shape (C_out, C_in*kh*kw)
    dcols = dY @ cache['weights']   # shape (N*out_h*out_w, C*kh*kw)
    dx = col2im(dcols, cache['x_shape'], cache['kernel_h'], cache['kernel_w'], cache['stride'], cache['padding']) 
    return dx

# Step 19 - conv2d_grad_weights
def conv2d_grad_weights(d_out, cache):
    # TODO: return dL/dW shaped (C_out, C_in, kH, kW) from d_out and the im2col cache.
    N,C_out,out_h,out_w = d_out.shape
    d_out = d_out.transpose(1,0,2,3).reshape(C_out, -1)
    d_W = d_out @ cache['cols']
    return d_W.reshape(C_out,-1,cache['kernel_h'],cache['kernel_w'])

# Step 20 - conv2d_grad_bias
def conv2d_grad_bias(d_out):
    # TODO: return a length C_out gradient by reducing d_out over batch and spatial axes
    return np.add.reduce(d_out, axis=(0,2,3))

# Step 21 - conv2d_backward
def conv2d_grad_input(d_out, cache):
    # TODO: backprop d_out through the conv input using col2im
    # d_out dims: (N,C_out,out_h,out_w)
    N,C_out,out_h,out_w = d_out.shape
    dY = d_out.transpose(0,2,3,1)
    dY = dY.reshape(-1,C_out)   # shape (N*out_h*out_w)
    # weights has shape (C_out, C_in*kh*kw)
    dcols = dY @ cache['weights']   # shape (N*out_h*out_w, C*kh*kw)
    dx = col2im(dcols, cache['x_shape'], cache['kernel_h'], cache['kernel_w'], cache['stride'], cache['padding']) 
    return dx

def conv2d_grad_weights(d_out, cache):
    # TODO: return dL/dW shaped (C_out, C_in, kH, kW) from d_out and the im2col cache.
    N,C_out,out_h,out_w = d_out.shape
    d_out = d_out.transpose(1,0,2,3).reshape(C_out, -1)
    d_W = d_out @ cache['cols']
    return d_W.reshape(C_out,-1,cache['kernel_h'],cache['kernel_w'])

def conv2d_grad_bias(d_out):
    # TODO: return a length C_out gradient by reducing d_out over batch and spatial axes
    return np.add.reduce(d_out, axis=(0,2,3))

def conv2d_backward(d_out, cache):
    # TODO: return (dx, dW, db) using the conv2d gradient helpers and the forward cache
    return (conv2d_grad_input(d_out, cache), conv2d_grad_weights(d_out, cache), conv2d_grad_bias(d_out))

# Step 22 - maxpool2d_forward
def output_spatial_size(input_size, kernel, stride, padding):
    # TODO: return the conv/pool output spatial dimension from input_size, kernel, stride, padding
    return int(np.floor((input_size + 2 * padding - kernel) / stride) + 1)

def maxpool2d_forward(x, kernel, stride):
    # TODO: run 2D max pooling and cache the in-window argmax of each output cell.
    N,C,H,W = x.shape
    out_h = output_spatial_size(H,kernel,stride,padding=0)
    out_w = output_spatial_size(W,kernel,stride,padding=0)
    out = np.empty((N,C,out_h,out_w), dtype=x.dtype)
    argmax = np.empty((N,C,out_h,out_w), dtype=int)
    for n in range(N):
        for c in range(C):
            for i in range(out_h):
                for j in range(out_w):
                    hs, ws = i*stride, j*stride 
                    out[n,c,i,j] = np.max(x[n,c,hs:hs+kernel,ws:ws+kernel])
                    argmax[n,c,i,j] = np.argmax(x[n,c,hs:hs+kernel,ws:ws+kernel])
    return (out, {
        'x_shape': x.shape,
        'argmax': argmax,
        'kernel': kernel,
        'stride': stride 
    })

# Step 23 - scatter_grad_window
import numpy as np

def scatter_grad_window(grad_value, argmax_index, kernel):
    # TODO: place grad_value at the argmax position within a (kernel, kernel) zero array.
    out = np.zeros((kernel,kernel))
    out[argmax_index//kernel,argmax_index%kernel] = grad_value
    return out

# Step 24 - maxpool2d_backward
def scatter_grad_window(grad_value, argmax_index, kernel):
    # TODO: place grad_value at the argmax position within a (kernel, kernel) zero array.
    out = np.zeros((kernel,kernel))
    out[argmax_index//kernel,argmax_index%kernel] = grad_value
    return out

def maxpool2d_backward(d_out, cache):
    # TODO: scatter each d_out value to the cached argmax position in its window
    out = np.zeros(cache['x_shape'], dtype=d_out.dtype)

    N,C,out_h,out_w = d_out.shape # NOTE: pooling does not change the number of channels
    s,k = cache['stride'], cache['kernel']
    for n in range(N):
        for c in range(C):
            for h in range(out_h):
                hs = h*s
                for w in range(out_w):
                    ws = w*s
                    window_grad = scatter_grad_window(d_out[n,c,h,w], cache['argmax'][n,c,h,w], k)
                    out[n,c,hs:hs+k,ws:ws+k] += window_grad
    return out

# Step 25 - relu_forward
def relu_forward(x):
    # TODO: Compute the elementwise ReLU and cache the input for backprop.
    out = np.copy(x)
    out[out<0] = 0
    return out, {
        'x': x
    }

# Step 26 - relu_backward
def relu_backward(d_out, cache):
    # TODO: mask the upstream gradient by the positive entries of the cached input.
    d_out[cache['x']<=0] = 0.0
    return d_out

# Step 27 - flatten_forward
def flatten_forward(x):
    # TODO: reshape a 4D feature map into a 2D batch matrix and cache the original shape
    N,C,H,W = x.shape
    return x.reshape(N,C*H*W), {
        'x_shape': x.shape
    }

# Step 28 - flatten_backward
import numpy as np

def flatten_backward(d_out, cache):
    # TODO: reshape the upstream gradient back to the original 4D feature map shape.
    N,C,H,W = cache['x_shape']
    return d_out.reshape(N,C,H,W)

# Step 29 - linear_forward
def linear_forward(x, weights, bias):
    # TODO: compute X @ W + b and cache the inputs needed for backprop.
    return x @ weights + bias, {
        'x': x,
        'weights': weights
    }

# Step 30 - linear_grad_input
import numpy as np

def linear_grad_input(d_out, cache):
    """Gradient of a linear layer w.r.t. its input X."""
    # TODO: return dL/dX given d_out (N, D_out) and cache['weights'] (D_in, D_out)
    return d_out @ cache['weights'].T

# Step 31 - linear_grad_weights
import numpy as np

def linear_grad_weights(x, dout):
    """Gradient of loss wrt linear-layer weights W of shape (D_in, D_out)."""
    # TODO: Compute the gradient of a linear layer's loss wrt its weight matrix W.
    return x.T @ dout

# Step 32 - linear_grad_bias
import numpy as np

def linear_grad_bias(dout):
    # TODO: Compute the bias gradient of a linear layer given upstream gradient dout.
    return np.add.reduce(dout, axis=0)

# Step 33 - linear_backward
def linear_grad_input(d_out, cache):
    """Gradient of a linear layer w.r.t. its input X."""
    # TODO: return dL/dX given d_out (N, D_out) and cache['weights'] (D_in, D_out)
    return d_out @ cache['weights'].T

def linear_grad_weights(x, dout):
    """Gradient of loss wrt linear-layer weights W of shape (D_in, D_out)."""
    # TODO: Compute the gradient of a linear layer's loss wrt its weight matrix W.
    return x.T @ dout

def linear_grad_bias(dout):
    # TODO: Compute the bias gradient of a linear layer given upstream gradient dout.
    return np.add.reduce(dout, axis=0)

def linear_backward(dout, cache):
    # TODO: combine input, weight, and bias gradients for a linear layer using the cache
    return linear_grad_input(dout, cache), linear_grad_weights(cache['x'], dout), linear_grad_bias(dout)

# Step 34 - softmax_cross_entropy_forward
def stable_softmax(logits):
    # TODO: Compute a numerically stable softmax row-wise over (N, C) logits.
    return exp_shifted(logits) / row_sum(exp_shifted(logits))

def cross_entropy_loss(probs, labels, eps=1e-12):
    # TODO: return the mean negative log-likelihood of the true-class probabilities
    out = -np.sum(np.log(np.clip(gather_true_class_probs(probs, labels), eps, None))) / probs.shape[0]
    return out if out != 0 else 0.0

def softmax_cross_entropy_forward(logits, y):
    # TODO: return the mean cross-entropy loss for logits (N, C) and integer labels y (N,).
    return cross_entropy_loss(stable_softmax(logits), y)

# Step 35 - softmax_cross_entropy_backward
def stable_softmax(logits):
    # TODO: Compute a numerically stable softmax row-wise over (N, C) logits.
    return exp_shifted(logits) / row_sum(exp_shifted(logits))

def one_hot(labels, num_classes):
    # TODO: convert integer labels into a (N, num_classes) one-hot float matrix
    out = np.zeros((labels.shape[0], num_classes), dtype=np.float64)
    for i in range(labels.shape[0]):
        for j in range(num_classes):
            if labels[i] == j:
                out[i][j] = 1.0
    return out

def softmax_cross_entropy_backward(logits, y):
    # TODO: return the fused softmax-cross-entropy gradient of shape (N, C).
    N,C = logits.shape
    return (stable_softmax(logits) - one_hot(y, C))/N

# Step 36 - sgd_step
import numpy as np

def sgd_step(param, grad, lr):
    # TODO: return the SGD-updated parameter array (param - lr * grad).
    return param - lr*grad

# Step 37 - adam_update_m
import numpy as np

def adam_update_m(m, grad, beta_one):
    # TODO: return the updated first moment estimate using beta_one and grad.
    return beta_one*m + (1-beta_one)*grad

# Step 38 - adam_update_v
import numpy as np

def adam_update_v(v, grad, beta_two):
    # TODO: return the updated Adam second moment estimate as an EMA of squared gradients.
    return beta_two*v + (1-beta_two)*(grad**2)

# Step 39 - adam_bias_correct
def adam_bias_correct(moment, beta, t):
    # TODO: return moment divided by (1 - beta**t) to undo Adam's zero-init bias.
    return moment / (1-beta**t)

# Step 40 - adam_param_step
import numpy as np

def adam_param_step(param, m_hat, v_hat, lr, eps):
    # TODO: apply one Adam parameter update using bias-corrected moments
    return param - lr*(m_hat/(np.sqrt(v_hat)+eps))

# Step 41 - adam_step
import numpy as np

def adam_update_m(m, grad, beta_one):
    # TODO: return the updated first moment estimate using beta_one and grad.
    return beta_one*m + (1-beta_one)*grad

def adam_update_v(v, grad, beta_two):
    # TODO: return the updated Adam second moment estimate as an EMA of squared gradients.
    return beta_two*v + (1-beta_two)*(grad**2)

def adam_bias_correct(moment, beta, t):
    # TODO: return moment divided by (1 - beta**t) to undo Adam's zero-init bias.
    return moment / (1-beta**t)

def adam_param_step(param, m_hat, v_hat, lr, eps):
    # TODO: apply one Adam parameter update using bias-corrected moments
    return param - lr*(m_hat/(np.sqrt(v_hat)+eps))

def adam_step(param, grad, m, v, t, lr, beta_one, beta_two, eps):
    new_m = adam_update_m(m, grad, beta_one)
    new_v = adam_update_v(v, grad, beta_two)
    m_hat = adam_bias_correct(new_m, beta_one, t)
    v_hat = adam_bias_correct(new_v, beta_two, t)
    new_param = adam_param_step(param, m_hat, v_hat, lr, eps)
    return new_param, new_m, new_v

# Step 42 - init_conv_layer
def he_init(shape, fan_in, seed):
    # TODO: sample a weight tensor from a normal distribution scaled by He std using the seed.
    np.random.seed(seed)
    std = he_std(fan_in)
    return np.random.normal(0, std, shape)

def init_zero_bias(length):
    # TODO: return a 1D float array of zeros with the given length.
    return np.zeros(length, dtype=np.float64)

def init_conv_layer(out_channels, in_channels, kernel_size, seed=0):
    # TODO: Build He-initialized weights and a zero bias for a single conv layer.
    return {
        'W': he_init((out_channels, in_channels, kernel_size, kernel_size), in_channels*kernel_size*kernel_size, seed),
        'b': init_zero_bias(out_channels)
    }

# Step 43 - init_linear_layer
def init_linear_layer(in_features, out_features, seed=0):
    # TODO: return {'W': He-init matrix (in_features, out_features), 'b': zero bias (out_features,)}
    return {
        'W': he_init((in_features, out_features), in_features, seed),
        'b': init_zero_bias(out_features)
    }

# Step 44 - init_lenet
def init_lenet(in_channels, num_classes, seed=0):
    # TODO: build conv1, conv2, fc1, fc2 with the right shapes and return them in a dict.
    return {
        'conv1': init_conv_layer(6, in_channels, 5, seed=0),
        'conv2': init_conv_layer(16, 6, 5, seed=1),
        'fc1': init_linear_layer(256, 120, seed=2),
        'fc2': init_linear_layer(120, num_classes, seed=3) 
    }

# Step 45 - forward_conv_block
def forward_conv_block(x, W, b, pool_size, stride, pad):
    # TODO: run conv2d -> relu -> maxpool2d and return (out, cache_dict)
    x, forward_cache = conv2d_forward(x, W, b, stride, pad)
    x, relu_cache = relu_forward(x)
    x, pool_cache = maxpool2d_forward(x, pool_size, pool_size)
    return x, {
        'conv_cache': forward_cache,
        'relu_cache': relu_cache,
        'pool_cache': pool_cache
    }

# Step 46 - forward_classifier_block
def forward_classifier_block(x, fc1, fc2):
    # TODO: run flatten -> linear -> relu -> linear and return logits plus a cache dict.
    x, flatten_cache = flatten_forward(x)
    x, fc1_cache = linear_forward(x, fc1['W'], fc1['b'])
    x, relu_cache = relu_forward(x)
    x, fc2_cache = linear_forward(x, fc2['W'], fc2['b'])
    return x, {
        'flatten_cache': flatten_cache,
        'fc1_cache': fc1_cache,
        'relu_cache': relu_cache,
        'fc2_cache': fc2_cache
    }

# Step 47 - lenet_forward
def lenet_forward(x, params):
    # TODO: run two conv blocks then the classifier block and return (logits, caches).
    x, block1_cache = forward_conv_block(x, params['conv1']['W'], params['conv1']['b'], 2, 1, 0)
    x, block2_cache = forward_conv_block(x, params['conv2']['W'], params['conv2']['b'], 2, 1, 0)
    x, classifier = forward_classifier_block(x, params['fc1'], params['fc2'])
    return x, {
        'block1': block1_cache,
        'block2': block2_cache,
        'classifier': classifier
    }

# Step 48 - backward_conv_block
def backward_conv_block(dout, cache):
    # TODO: backprop dout through the cached pool, relu, and conv layers in reverse order.
    return conv2d_backward(relu_backward(maxpool2d_backward(dout, cache['pool_cache']), cache['relu_cache']), cache['conv_cache'])

# Step 49 - backward_classifier_block
def backward_classifier_block(dlogits, cache):
    # TODO: backprop through fc2 -> relu -> fc1 -> flatten using the cached values
    x, dW2, db2 = linear_backward(dlogits, cache['fc2_cache'])
    x = relu_backward(x, cache['relu_cache'])
    x, dW1, db1 = linear_backward(x, cache['fc1_cache'])
    x = flatten_backward(x, cache['flatten_cache'])
    return {
        'dx': x,
        'fc1': {
            'dW': dW1,
            'db': db1
        },
        'fc2': {
            'dW': dW2,
            'db': db2 
        }
    }

# Step 50 - lenet_backward
def lenet_backward(dlogits, caches):
    # TODO: walk classifier and conv block caches in reverse to assemble all gradients
    classifier_dict = backward_classifier_block(dlogits, caches['classifier'])
    dx, dW2, db2 = backward_conv_block(classifier_dict['dx'], caches['block2'])
    dx, dW1, db1 = backward_conv_block(dx, caches['block1'])
    return {
        'conv1': {
            'dW': dW1,
            'db': db1
        },
        'conv2': {
            'dW': dW2,
            'db': db2
        },
        'fc1': {
            'dW': classifier_dict['fc1']['dW'],
            'db': classifier_dict['fc1']['db'],
        },
        'fc2': {
            'dW': classifier_dict['fc2']['dW'],
            'db': classifier_dict['fc2']['db']
        }
    }

# Step 51 - lenet_predict
def lenet_predict(x, params):
    # TODO: Return the argmax class index per sample from a LeNet forward pass.
    x, caches = lenet_forward(x, params)
    return argmax_rows(x)

# Step 52 - build_synthetic_image_dataset
def build_synthetic_image_dataset(num_samples, num_classes, image_size, in_channels=1, seed=0):
    # TODO: Return (x, y) for a reproducible synthetic NCHW image dataset.
    rng = np.random.default_rng(seed)
    y = rng.integers(num_classes, size=num_samples)
    x = rng.standard_normal((num_samples, in_channels, image_size, image_size))
    for j in range(num_classes):
        x[y==j,:,:,:] += (j-(num_classes-1))/2
    return x,y

# Step 53 - shuffle_indices
import numpy as np

def shuffle_indices(n, seed=0):
    rng = np.random.RandomState(seed)
    return rng.permutation(n)

# Step 54 - train_test_split
def train_test_split(x, y, test_fraction=0.2, seed=0):
    n = x.shape[0]
    idx = shuffle_indices(n, seed=seed)
    n_test = int(n * test_fraction)
    test_idx, train_idx = idx[:n_test], idx[n_test:]
    return x[train_idx], y[train_idx], x[test_idx], y[test_idx]

# Step 55 - iterate_minibatches
def iterate_minibatches(x, y, batch_size, seed=0):
    n = x.shape[0]
    idxs = shuffle_indices(n, seed=seed)
    for start in range(0, n - batch_size + 1, batch_size):
        batch = idxs[start:start + batch_size]
        yield x[batch], y[batch]

# Step 56 - train_step
def train_step(params, opt_state, xb, yb, lr, beta_one, beta_two, eps, step):
    logits, caches = lenet_forward(xb, params)
    loss = softmax_cross_entropy_forward(logits, yb)
    dlogits = softmax_cross_entropy_backward(logits, yb)
    grads = lenet_backward(dlogits, caches)

    new_params, new_opt_state = {}, {}
    for layer in params:
        new_params[layer], new_opt_state[layer] = {}, {}
        for pname in params[layer]:
            p, m, v = adam_step(
                params[layer][pname], grads[layer]['d'+pname],
                opt_state[layer][pname]['m'], opt_state[layer][pname]['v'],
                step, lr, beta_one, beta_two, eps,
            )
            new_params[layer][pname] = p
            new_opt_state[layer][pname] = {'m': m, 'v': v}
    return new_params, new_opt_state, loss

# Step 57 - train_one_epoch
def train_one_epoch(params, opt_state, x, y, batch_size, lr, beta_one, beta_two, eps, step_counter, seed=0):
    # TODO: iterate minibatches and apply one train_step per batch, tracking losses and step_counter.
    losses = []
    for xb,yb in iterate_minibatches(x,y,batch_size,seed):
        step_counter += 1
        params, opt_state, loss = train_step(params, opt_state, xb, yb, lr, beta_one, beta_two, eps, step_counter)
        
        losses.append(loss)
    return params, opt_state, step_counter, losses

# Step 58 - train_loop
def train_loop(params, x_train, y_train, num_epochs, batch_size, lr=1e-3,
               beta_one=0.9, beta_two=0.999, eps=1e-8, seed=0):
    opt_state = {}
    for layer in params:
        opt_state[layer] = {}
        for pname in params[layer]:
            opt_state[layer][pname] = {'m': 0.0, 'v': 0.0}
    step = 0
    loss_history = []
    for epoch in range(num_epochs):
        params, opt_state, step, losses = train_one_epoch(
            params, opt_state, x_train, y_train, batch_size,
            lr, beta_one, beta_two, eps, step, seed + epoch,
        )
        loss_history.extend(losses)
    return params, loss_history

# Step 59 - evaluate (not yet solved)
# TODO: implement

