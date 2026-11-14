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

# Step 22 - maxpool2d_forward (not yet solved)
# TODO: implement

# Step 23 - scatter_grad_window (not yet solved)
# TODO: implement

# Step 24 - maxpool2d_backward (not yet solved)
# TODO: implement

# Step 25 - relu_forward (not yet solved)
# TODO: implement

# Step 26 - relu_backward (not yet solved)
# TODO: implement

# Step 27 - flatten_forward (not yet solved)
# TODO: implement

# Step 28 - flatten_backward (not yet solved)
# TODO: implement

# Step 29 - linear_forward (not yet solved)
# TODO: implement

# Step 30 - linear_grad_input (not yet solved)
# TODO: implement

# Step 31 - linear_grad_weights (not yet solved)
# TODO: implement

# Step 32 - linear_grad_bias (not yet solved)
# TODO: implement

# Step 33 - linear_backward (not yet solved)
# TODO: implement

# Step 34 - softmax_cross_entropy_forward (not yet solved)
# TODO: implement

# Step 35 - softmax_cross_entropy_backward (not yet solved)
# TODO: implement

# Step 36 - sgd_step (not yet solved)
# TODO: implement

# Step 37 - adam_update_m (not yet solved)
# TODO: implement

# Step 38 - adam_update_v (not yet solved)
# TODO: implement

# Step 39 - adam_bias_correct (not yet solved)
# TODO: implement

# Step 40 - adam_param_step (not yet solved)
# TODO: implement

# Step 41 - adam_step (not yet solved)
# TODO: implement

# Step 42 - init_conv_layer (not yet solved)
# TODO: implement

# Step 43 - init_linear_layer (not yet solved)
# TODO: implement

# Step 44 - init_lenet (not yet solved)
# TODO: implement

# Step 45 - forward_conv_block (not yet solved)
# TODO: implement

# Step 46 - forward_classifier_block (not yet solved)
# TODO: implement

# Step 47 - lenet_forward (not yet solved)
# TODO: implement

# Step 48 - backward_conv_block (not yet solved)
# TODO: implement

# Step 49 - backward_classifier_block (not yet solved)
# TODO: implement

# Step 50 - lenet_backward (not yet solved)
# TODO: implement

# Step 51 - lenet_predict (not yet solved)
# TODO: implement

# Step 52 - build_synthetic_image_dataset (not yet solved)
# TODO: implement

# Step 53 - shuffle_indices (not yet solved)
# TODO: implement

# Step 54 - train_test_split (not yet solved)
# TODO: implement

# Step 55 - iterate_minibatches (not yet solved)
# TODO: implement

# Step 56 - train_step (not yet solved)
# TODO: implement

# Step 57 - train_one_epoch (not yet solved)
# TODO: implement

# Step 58 - train_loop (not yet solved)
# TODO: implement

# Step 59 - evaluate (not yet solved)
# TODO: implement

