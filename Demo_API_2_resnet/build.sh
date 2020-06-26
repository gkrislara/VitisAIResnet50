#!/bin/bash

CXX=${CXX:-g++}
$CXX -std=c++11 -O3 -I. -o resnet50_tf_532020 resnet50_dpu.cpp -ln2cube -lhineon -lopencv_imgcodecs -lopencv_highgui -lopencv_imgproc -lopencv_core -pthread -lxrt_core