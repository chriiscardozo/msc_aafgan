#!/bin/bash

# COUNT=$(ps aux | grep mnist_condgan_main | wc -l)
COUNT=$(ps aux | grep celeba_dcgan_main | wc -l)
if [ "$COUNT" -le 1 ]
then
    cd /home/christian/msc_aafgan/
    # source /home/christian/anaconda3/bin/activate base && python3 -m Main.mnist_condgan_main condgan_mlp
    source /home/christian/anaconda3/bin/activate base && python3 -m Main.celeba_dcgan_main celeba_dcgan_henormal
fi
