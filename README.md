# Master's degree code: AAF-GAN (Adaptive Activation Functions GAN)

Code repository for my master's degree dissertation about applying hyperbolic-based adptive activation functions in GANs (Generative Adversarial Nets). Master dregree granted by [PESC/COPPE](https://www.cos.ufrj.br/) at the Federal University of Rio de Janeiro ([UFRJ](http://ufrj.br/)).

## Proposals

- AAF-GAN: GAN architecture using adaptive activation functions. We validate the new architecture by applying activation functions based on the hyperbolic penalization method.
- MiDA: a novel adaptive activation function using an adaptive symmetric bi-hyperbolic (BHSA) activation function to apply self-gating property by replacing the *tanh* in the oringal Mish non-adaptive function.
- BHANA: matematical proposal for normalize the adaptive asymmetric bi-hyperbolic (BHAA) activation function and solve the range limits problem.

## Cite this work

```
to do
```

## Dissertation's Abstract

Generative adversarial networks (GANs) are a generative model architecture proposed a few years ago where two neural networks train against each other. The generator network tries to produce samples similar to real data with no direct contact with them, and the discriminator network attempts to maximize the accuracy of predicting which data is fake generated and legit from the dataset. Adaptive activation functions are an approach yet not explored in GANs that could improve their performance. Specifically, functions based on the hyperbolic penalization technique indicate that convergence can be improved in the multi-layer network's training's early epochs.

In this work, we propose a new approach to GANs networks by applying adaptive activation functions based on hyperbolic penalization to analyze the impact on the generated samples quality, considering the image generation task. We also propose a new version using adaptive parameters for the Mish function by changing its formula to use a hyperbolic function to self-gating. Finally, we explore the theoretical limitation of the Adaptive Asymmetric Bi-hyperbolic function due to non-limited boundaries by proposing a new normalized formulation. We found strategies resulting in better performance for early convergence for GANs. Also, we validate that the MiDA function can, at minimum, improve the convergence compared to Mish performance. The discoveries were compiled into guidelines that can be used as the base for applying adaptive functions to GANs.

---

## Setup

This section describes how to setup your environment to run this code. The procedures below are based on the usage of AWS EC2 instances, but in general the steps can adapted for your personal use case.

##### 1. Instance type

* **Machine Image**: Deep Learning AMI (Ubuntu 16.04)
* **Type**: suggestion is use *p2.xlarge* for **mnist_condgan** experiments and *p3.2xlarge* for **celeba_dcgan**, based on the best cost/benefit rate of epoch running time per hourly price
* **Spot instance**: supported using **persistent request** with **stop behavior**
* **Storage size**: 'instance default' + 50 GB per model included
* **Tags (optional)**:
  * Using model name structur. For example to run only the BHSA in Discriminator Network model: `NAME=output_Dis_BHSA`
  * Using structure parameters to define what models must be executed
```
GEN_MODELS=default,BHAA,BHSA,MiDA,Mish
DIS_MODELS=default,BHAA,BHSA
DIS_SHRELU=0,1
GEN_SHRELU=0,1
```

**Warning note:** Using tags in EC2 instances with the keys `NAME`, `GEN_MODELS`, `DIS_MODELS`, `DIS_SHRELU`, `GEN_SHRELU` will override the JSON configuration file for model selecting. 

##### 2. Code setup

- Repository download

```
source activate pytorch_p36
git clone https://github.com/chriiscardozo/msc_aafgan.git
sudo ln -s /home/ubuntu/msc_aafgan /msc_aafgan
chmod +x /msc_aafgan/Main/startup_aws.sh
```

- Additional dependencies setup

```
# Skip the pip3 install below if you have already PyTorch installed

pip3 install torch torchvision
```

```
sudo apt-get update
pip3 install torchsummary pytorch_fid twilio awscli
sudo apt-get -y install s3fs cloud-utils
```

- S3 bucket locally mounted 

```
# replace "BLAH:BLAH" with your access and secret keys from AWS
# replace YOUR-S3-BUCKET with your bucket name in S3

CLIP="BLAH:BLAH"
echo $CLIP > ~/.passwd-s3fs
sudo chmod 600 ~/.passwd-s3fs
mkdir ~/s3-drive
s3fs YOUR-S3-BUCKET ~/s3-drive -o umask=0007,uid=$UID
ls -l ~/s3-drive
```

- AWS utils configuration (optional)

```
aws configure

* AWS Access Key ID [None]: BLAH
* AWS Secret Access Key [None]: BLAH
* Default region name [None]: eu-west-1
* Default output format [None]: 

```

`eu-west-1` is Ireland AWS region. Replace according to your region.

- Extra: Examples of commands using the AWS utils

```
# Getting instance id
ec2metadata --instance-id

# Stopping instance
aws ec2 stop-instances --instance-ids "INSTANCE-ID"
```

##### 3. CelebA dataset setup

CelebA dataset may be unnavailable by using PyTorch automatically download feature. If so, try to download it directly from the [official website](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) and keep a copy in your S3 bucket for further downloads.

Assuming you have downloaded the CelebA dataset and you saved it in your S3 bucket with the location `s3://YOUR-S3-BUCKET/celeba.zip`, run:

```
mkdir ~/msc_aafgan/data/
aws s3 cp s3://YOUR-S3-BUCKET/celeba.zip ~/msc_aafgan/data/
unzip ~/msc_aafgan/data/celeba.zip -d ~/msc_aafgan/data/
unzip ~/msc_aafgan/data/celeba/img_align_celeba.zip -d ~/msc_aafgan/data/celeba/
```

##### 4. cron configuration (optional)

This is useful specially if we are using spot instances in AWS.

```
* * * * * /msc_aafgan/Main/startup_aws.sh >> /tmp/cron.log 2>&1
```

##### 5. The rsync to validation host optional)

Save the script below and execute it to keep local and experiment machine code in sync without need to commit. It is helpful for testing purposes.

```
ssh -N -L 9876:END_HOST:END_HOST_PORT -p MIDDLE_HOST_PORT USERNAME@MIDDLE_HOST &
echo "starting syncing"
fswatch MY_LOCAL_PATH_TO_AAFGAN/msc_aafgan | (while read; do rsync -auve "ssh -p 9876 -i ssh_keys/RSA_KEY" MY_LOCAL_PATH_TO_AAFGAN/msc_aafgan/ USERNAME@localhost:/DESTINATION_PATH/msc_aafgan; done)
```
