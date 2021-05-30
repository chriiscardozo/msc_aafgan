#!/bin/bash

COUNT=$(ps aux | grep celeba_dcgan_main | wc -l)

if [ "$COUNT" -le 1 ]
then
    cd /msc_aafgan/
    source /home/ubuntu/anaconda3/bin/activate pytorch_p36 && python3 -m Main.celeba_dcgan_main celeba_dcgan
fi
