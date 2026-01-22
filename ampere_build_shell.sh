#!/usr/bin/env bash
set -e

cd fbgemm_gpu/
git submodule update --init --recursive
pip install -r requirements.txt

# Install HSTU-Ampere
export HSTU_DISABLE_BACKWARD=FALSE; \
export HSTU_DISABLE_LOCAL=FALSE; \
export HSTU_DISABLE_CAUSAL=FALSE; \
export HSTU_DISABLE_CONTEXT=TRUE; \
export HSTU_DISABLE_TARGET=TRUE; \
export HSTU_DISABLE_ARBITRARY=TRUE; \
export HSTU_ARBITRARY_NFUNC=3; \
export HSTU_DISABLE_RAB=FALSE; \
export HSTU_DISABLE_DRAB=TRUE; \
export HSTU_DISABLE_BF16=FALSE; \
export HSTU_DISABLE_FP16=TRUE; \
export HSTU_DISABLE_HDIM32=TRUE; \
export HSTU_DISABLE_HDIM64=TRUE; \
export HSTU_DISABLE_HDIM128=FALSE; \
export HSTU_DISABLE_HDIM256=TRUE; \
export HSTU_DISABLE_DETERMINISTIC=TRUE; \
export HSTU_DISABLE_86OR89=TRUE; \
python setup.py install --build-target=hstu -DTORCH_CUDA_ARCH_LIST="8.0"

