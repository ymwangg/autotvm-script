#!/bin/env python
import os, sys
import time
import tvm
from tvm import topi
import numpy as np
import shutil
from tvm import autotvm
import pickle


try:
    log_file_name = sys.argv[1]
    records = []
    for r in autotvm.record.load_from_file(log_file_name):
        records.append(r)
    dir_name = "{}.benchmark".format(log_file_name)
    if os.path.exists(dir_name):
        shutil.rmtree(dir_name)
    os.mkdir(dir_name)
except Exception as e:
    raise RuntimeError(e)

i = 0 
for (inp,res) in records:
    target, task, config = inp 
    print(target)
    try:
        with tvm.target.create(target):
            s, args = task.instantiate(config)
            with tvm.ir.transform.PassContext(opt_level=3, disabled_pass=['tir.UnrollLoop'], config={'tir.disable_vectorize': True}):
                func = tvm.build(s, args, target=target)
                func.export_library(os.path.join(dir_name, "func.{}.so".format(i)))
            with open(os.path.join(dir_name, "args.{}.pickle".format(i)),'wb') as f_args:
                pickle.dump(args, f_args)
    except Exception as e:
        print(e)
    i += 1
