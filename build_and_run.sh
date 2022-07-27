#!/bin/bash
set -e

rm -rf build
mkdir build
cd build
cmake -DCMAKE_VERBOSE_MAKEFILE=ON ..
make

./conv2d_relu_pool2d
