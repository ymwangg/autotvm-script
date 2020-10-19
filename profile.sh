#!/bin/bash
echo $@
sudo PATH=$PATH TVM_HOME=$TVM_HOME PYTHONPATH=$PYTHONPATH nvprof --events shared_ld_bank_conflict,shared_st_bank_conflict,shared_ld_transactions,shared_st_transactions $@
