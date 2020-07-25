#! /usr/bin/env bash

# install environment for this project

# 生成可用中国镜像站列表
# sudo pacman-mirrors -i -c China -m rank
# add ustc archlinuxcn mirror. add at end of /etc/pacman.conf
# [archlinuxcn]
# SigLevel = Optional TrustedOnly
# Server = https://mirrors.ustc.edu.cn/archlinuxcn/$arch
# 刷新缓存
# sudo pacman -Syy
# 导入GPG key
# sudo pacman -S archlinuxcn-keyring

yay -S cmake
yay -S google-glog gflags
yay -S cuda cudnn
yay -S opencv-cuda
# sudo ln -s /opt/cuda /usr/local/cuda
yay -S libtorch