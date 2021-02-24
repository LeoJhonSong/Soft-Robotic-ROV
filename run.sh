#!/usr/bin/env bash
./utils/js2key &
./build/Soft-Robotics-ROV \
--RUAS=0 \
--K=100 \
--R=50 \
--NET_PHASE=2 \
--MODE=-1 \
--SSD_DIM=320 \
--NETG_DIM=256 \
--TUB=true \
--TRACK=true \
--RECORD=false
