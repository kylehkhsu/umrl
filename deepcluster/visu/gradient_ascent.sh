# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
#!/bin/bash

MODEL='/home/jabreezus/clones/deepcluster/vizdoom1/checkpoint.pth.tar'
ARCH='resnet18'
EXP='/home/jabreezus/clones/deepcluster/vizdoom1/'
CONV=6

python gradient_ascent.py --model ${MODEL} --exp ${EXP} --conv ${CONV} --arch ${ARCH}
