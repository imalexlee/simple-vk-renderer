#!/bin/sh
cmake -DCMAKE_BUILD_TYPE=Release -DGLFW_BUILD_DOCS=OFF -S ../ -B ../out/release
