#!/bin/env python
for i in {0..21167}
do
    profile.sh ./benchmark.py resnet-18.log.tmp.benchmark/func.$i.so resnet-18.log.tmp.benchmark/args.$i.pickle 2> resnet-18.log.tmp.benchmark/$i.log
done
