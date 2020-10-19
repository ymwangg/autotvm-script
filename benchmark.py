#!/bin/env python
import tvm
from tvm import topi
from tvm.topi.util import get_const_tuple
import sys
import pickle
import numpy as np
import time


try:
    func_file_name = sys.argv[1]
    args_file_name = sys.argv[2]

    func = tvm.runtime.load_module(func_file_name)
    with open(args_file_name, 'rb') as f_args:
        args = pickle.load(f_args)
    ctx = tvm.gpu(0)
    a = tvm.nd.array(np.random.uniform(0.0, 255.0, size=get_const_tuple(args[0].shape)).astype(args[0].dtype), ctx)
    b = tvm.nd.array(np.random.uniform(0.0, 255.0, size=get_const_tuple(args[1].shape)).astype(args[1].dtype), ctx)
    c = tvm.nd.array(np.zeros(get_const_tuple(args[2].shape), dtype=args[2].dtype), ctx)
    time_f = func.time_evaluator(func.entry_name, ctx, number=10, repeat=1, min_repeat_ms=1000)
    t = time_f(a,b,c).mean
    print(t*1000)
except Exception as e:
    raise RuntimeError(e)
