#!/usr/bin/env bash
./utils/js2key &
./build/Soft-Robotics-ROV \
--RUAS=0 \
--K=100 \
--R=50 \
--NET_PHASE=2 \
--MODE=-2 \
--SSD_DIM=320 \
--NETG_DIM=256 \
--TUB=1 \
--UART=true \
--WITH_ROV=true \
--TRACK=true \
--RECORD=false
