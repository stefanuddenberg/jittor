# ***************************************************************
# Copyright (c) 2020 Jittor. Authors:
#   Dun Liang <randonlang@gmail.com>.
#   Wenyang Zhou <576825820@qq.com>
#
# All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
import jittor as jt
import numpy as np
from collections.abc import Sequence
from pdb import set_trace as st

def repeat(x, *shape):
    if len(shape) == 1 and isinstance(shape[0], Sequence):
        shape = shape[0]
    len_x_shape = len(x.shape)
    len_shape = len(shape)
    x_shape = x.shape
    rep_shape = shape
    if len_x_shape < len_shape:
        x_shape = (len_shape - len_x_shape) * [1] + x.shape
        x = x.broadcast(x_shape)
    elif len_x_shape > len_shape:
        rep_shape = (len_x_shape - len_shape) * [1] + shape
    tar_shape = (np.array(x_shape) * np.array(rep_shape)).tolist()
    dims = []
    for i in range(len(tar_shape)): dims.append(f"i{i}%{x_shape[i]}")
    return x.reindex(tar_shape, dims)
jt.Var.repeat = repeat

def chunk(x, chunks, dim=0):
    l = x.shape[dim]
    res = []
    if l <= chunks:
        for i in range(l):
            res.append(x[(slice(None,),)*dim+([i,],)])
    else:
        nums = (l-1) // chunks + 1
        for i in range(chunks-1):
            res.append(x[(slice(None,),)*dim+(slice(i*nums,(i+1)*nums),)])
        if (i+1)*nums < l:
            res.append(x[(slice(None,),)*dim+(slice((i+1)*nums,None),)])
    return res
jt.Var.chunk = chunk

def stack(x, dim=0):
    assert isinstance(x, list)
    assert len(x) >= 2
    res = [x_.unsqueeze(dim) for x_ in x]
    return jt.contrib.concat(res, dim=dim)
jt.Var.stack = stack

def flip(x, dim=0):
    assert isinstance(dim, int)
    tar_dims = []
    for i in range(len(x.shape)):
        if i == dim:
            tar_dims.append(f"{x.shape[dim]-1}-i{i}")
        else:
            tar_dims.append(f"i{i}")
    return x.reindex(x.shape, tar_dims)
jt.Var.flip = flip
