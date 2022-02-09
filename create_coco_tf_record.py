警告：您目前连接的是 GPU 运行时，但是没有使用 GPU。
vit
vit_
See code at https://github.com/google-research/vision_transformer/

See papers at

Vision Transformer: https://arxiv.org/abs/2010.11929
MLP-Mixer: https://arxiv.org/abs/2105.01601
How to train your ViT: https://arxiv.org/abs/2106.10270
When Vision Transformers Outperform ResNets without Pretraining or Strong Data Augmentations: https://arxiv.org/abs/2106.01548
This Colab allows you to run the JAX implementation of the Vision Transformer.

If you just want to load a pre-trained checkpoint from a large repository and directly use it for inference, you probably want to go the other Colab

https://colab.sandbox.google.com/github/google-research/vision_transformer/blob/linen/vit_jax_augreg.ipynb

[2]
%%bash
pip install oss2
pip install tensorflow-object-detection-api
[3]
26 秒
import os
from google.colab import drive
import oss2

drive.mount('/content/drive',force_remount=True)
os.chdir("/content/drive/MyDrive/Colab Notebooks")
Mounted at /content/drive
[4]
1 分钟
%%bash
# git clone https://github.com/missingstuffedbun/detr.git
# git clone https://github.com/tensorflow/models.git
# git clone --depth=1 https://github.com/google-research/vision_transformer
pip install -qr vision_transformer/vit_jax/requirements.txt
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
yellowbrick 1.3.post1 requires numpy<1.20,>=1.16.0, but you have numpy 1.21.5 which is incompatible.
datascience 0.10.6 requires folium==0.2.1, but you have folium 0.8.3 which is incompatible.
albumentations 0.1.12 requires imgaug<0.2.7,>=0.2.5, but you have imgaug 0.2.9 which is incompatible.
[5]
0 秒
# %%bash
# gsutil ls -lh gs://vit_models/imagenet*
# gsutil ls -lh gs://vit_models/sam
# gsutil ls -lh gs://mixer_models/*
[6]
0 秒
# # Download a pre-trained model.

# # Note: you can really choose any of the above, but this Colab has been tested
# # with the models of below selection...
# model_name = 'Mixer-B_16'  #@param ["ViT-B_32", "Mixer-B_16"]

# if model_name.startswith('ViT'):
#   ![ -e "$model_name".npz ] || gsutil cp gs://vit_models/imagenet21k/"$model_name".npz .
# if model_name.startswith('Mixer'):
#   ![ -e "$model_name".npz ] || gsutil cp gs://mixer_models/imagenet21k/"$model_name".npz .

# import os
# assert os.path.exists(f'{model_name}.npz')
[7]
0 秒
os.environ['DATAPATH'] = '/content'
[8]
7 秒
%%bash
mkdir -p /content/input
mkdir -p /content/working
mkdir -p /content/input/sarship
mkdir -p /content/input/sarship/train
mkdir -p /content/input/sarship/val
mkdir -p /content/input/sarship/test
tar -C /content/input/sarship/train -xf ./sarship/train.tar
tar -C /content/input/sarship/val -xf ./sarship/val.tar
tar -C /content/input/sarship/test -xf ./sarship/test.tar
mkdir /content/input/tf
[9]
7 秒
ali_key = 'LTAI5tDZzxHjQET1nQa1z1Pg'
ali_token = 'kYoV38UUEHK5ha2UTiVa0bm6s38aMF'

auth = oss2.Auth(ali_key, ali_token)
bucket = oss2.Bucket(auth, 'https://oss-cn-shenzhen.aliyuncs.com', 'missingstuffedbun-shelter227')

bucket.get_object_to_file('custom_train.json', '{}/working/custom_train.json'.format(os.environ['DATAPATH']))
bucket.get_object_to_file('custom_val.json', '{}/working/custom_val.json'.format(os.environ['DATAPATH']))
bucket.get_object_to_file('custom_test.json', '{}/working/custom_test.json'.format(os.environ['DATAPATH']))
<oss2.models.GetObjectResult at 0x7f89394a3ed0>
[19]
2 秒
! rm -rf detr && git clone https://github.com/missingstuffedbun/detr.git
Cloning into 'detr'...
remote: Enumerating objects: 296, done.
remote: Counting objects: 100% (12/12), done.
remote: Compressing objects: 100% (12/12), done.
remote: Total 296 (delta 6), reused 0 (delta 0), pack-reused 284
Receiving objects: 100% (296/296), 12.86 MiB | 17.46 MiB/s, done.
Resolving deltas: 100% (162/162), done.
[31]
22 秒
%%bash
python detr/create_coco_tf_record.py --logtostderr \
--train_image_dir=$DATAPATH/input/sarship/train/train \
--val_image_dir=$DATAPATH/input/sarship/val/val \
--test_image_dir=$DATAPATH/input/sarship/test/test \
--train_annotations_file=$DATAPATH/working/custom_train.json \
--val_annotations_file=$DATAPATH/working/custom_val.json \
--test_annotations_file=$DATAPATH/working/custom_test.json \
--output_dir=$DATAPATH/input/tf
INFO:tensorflow:Found groundtruth annotations. Building annotations index.
I0209 07:29:29.604328 140107823540096 create_coco_tf_record.py:209] Found groundtruth annotations. Building annotations index.
INFO:tensorflow:0 images are missing annotations.
I0209 07:29:29.628294 140107823540096 create_coco_tf_record.py:222] 0 images are missing annotations.
INFO:tensorflow:On image 0 of 30674
I0209 07:29:29.628699 140107823540096 create_coco_tf_record.py:227] On image 0 of 30674
INFO:tensorflow:On image 100 of 30674
I0209 07:29:29.693423 140107823540096 create_coco_tf_record.py:227] On image 100 of 30674
INFO:tensorflow:On image 200 of 30674
I0209 07:29:29.739282 140107823540096 create_coco_tf_record.py:227] On image 200 of 30674
INFO:tensorflow:On image 300 of 30674
I0209 07:29:29.788515 140107823540096 create_coco_tf_record.py:227] On image 300 of 30674
INFO:tensorflow:On image 400 of 30674
I0209 07:29:29.836873 140107823540096 create_coco_tf_record.py:227] On image 400 of 30674
INFO:tensorflow:On image 500 of 30674
I0209 07:29:29.885482 140107823540096 create_coco_tf_record.py:227] On image 500 of 30674
INFO:tensorflow:On image 600 of 30674
I0209 07:29:29.933938 140107823540096 create_coco_tf_record.py:227] On image 600 of 30674
INFO:tensorflow:On image 700 of 30674
I0209 07:29:29.982747 140107823540096 create_coco_tf_record.py:227] On image 700 of 30674
INFO:tensorflow:On image 800 of 30674
I0209 07:29:30.032539 140107823540096 create_coco_tf_record.py:227] On image 800 of 30674
INFO:tensorflow:On image 900 of 30674
I0209 07:29:30.083466 140107823540096 create_coco_tf_record.py:227] On image 900 of 30674
INFO:tensorflow:On image 1000 of 30674
I0209 07:29:30.131664 140107823540096 create_coco_tf_record.py:227] On image 1000 of 30674
INFO:tensorflow:On image 1100 of 30674
I0209 07:29:30.180252 140107823540096 create_coco_tf_record.py:227] On image 1100 of 30674
INFO:tensorflow:On image 1200 of 30674
I0209 07:29:30.228890 140107823540096 create_coco_tf_record.py:227] On image 1200 of 30674
INFO:tensorflow:On image 1300 of 30674
I0209 07:29:30.277408 140107823540096 create_coco_tf_record.py:227] On image 1300 of 30674
INFO:tensorflow:On image 1400 of 30674
I0209 07:29:30.326297 140107823540096 create_coco_tf_record.py:227] On image 1400 of 30674
INFO:tensorflow:On image 1500 of 30674
I0209 07:29:30.376172 140107823540096 create_coco_tf_record.py:227] On image 1500 of 30674
INFO:tensorflow:On image 1600 of 30674
I0209 07:29:30.424551 140107823540096 create_coco_tf_record.py:227] On image 1600 of 30674
INFO:tensorflow:On image 1700 of 30674
I0209 07:29:30.473644 140107823540096 create_coco_tf_record.py:227] On image 1700 of 30674
INFO:tensorflow:On image 1800 of 30674
I0209 07:29:30.522514 140107823540096 create_coco_tf_record.py:227] On image 1800 of 30674
INFO:tensorflow:On image 1900 of 30674
I0209 07:29:30.570660 140107823540096 create_coco_tf_record.py:227] On image 1900 of 30674
INFO:tensorflow:On image 2000 of 30674
I0209 07:29:30.625664 140107823540096 create_coco_tf_record.py:227] On image 2000 of 30674
INFO:tensorflow:On image 2100 of 30674
I0209 07:29:30.678481 140107823540096 create_coco_tf_record.py:227] On image 2100 of 30674
INFO:tensorflow:On image 2200 of 30674
I0209 07:29:30.726355 140107823540096 create_coco_tf_record.py:227] On image 2200 of 30674
INFO:tensorflow:On image 2300 of 30674
I0209 07:29:30.773896 140107823540096 create_coco_tf_record.py:227] On image 2300 of 30674
INFO:tensorflow:On image 2400 of 30674
I0209 07:29:30.822477 140107823540096 create_coco_tf_record.py:227] On image 2400 of 30674
INFO:tensorflow:On image 2500 of 30674
I0209 07:29:30.871159 140107823540096 create_coco_tf_record.py:227] On image 2500 of 30674
INFO:tensorflow:On image 2600 of 30674
I0209 07:29:30.919014 140107823540096 create_coco_tf_record.py:227] On image 2600 of 30674
INFO:tensorflow:On image 2700 of 30674
I0209 07:29:30.969668 140107823540096 create_coco_tf_record.py:227] On image 2700 of 30674
INFO:tensorflow:On image 2800 of 30674
I0209 07:29:31.022307 140107823540096 create_coco_tf_record.py:227] On image 2800 of 30674
INFO:tensorflow:On image 2900 of 30674
I0209 07:29:31.073893 140107823540096 create_coco_tf_record.py:227] On image 2900 of 30674
INFO:tensorflow:On image 3000 of 30674
I0209 07:29:31.125197 140107823540096 create_coco_tf_record.py:227] On image 3000 of 30674
INFO:tensorflow:On image 3100 of 30674
I0209 07:29:31.180253 140107823540096 create_coco_tf_record.py:227] On image 3100 of 30674
INFO:tensorflow:On image 3200 of 30674
I0209 07:29:31.235112 140107823540096 create_coco_tf_record.py:227] On image 3200 of 30674
INFO:tensorflow:On image 3300 of 30674
I0209 07:29:31.288720 140107823540096 create_coco_tf_record.py:227] On image 3300 of 30674
INFO:tensorflow:On image 3400 of 30674
I0209 07:29:31.340614 140107823540096 create_coco_tf_record.py:227] On image 3400 of 30674
INFO:tensorflow:On image 3500 of 30674
I0209 07:29:31.393158 140107823540096 create_coco_tf_record.py:227] On image 3500 of 30674
INFO:tensorflow:On image 3600 of 30674
I0209 07:29:31.442740 140107823540096 create_coco_tf_record.py:227] On image 3600 of 30674
INFO:tensorflow:On image 3700 of 30674
I0209 07:29:31.491904 140107823540096 create_coco_tf_record.py:227] On image 3700 of 30674
INFO:tensorflow:On image 3800 of 30674
I0209 07:29:31.539294 140107823540096 create_coco_tf_record.py:227] On image 3800 of 30674
INFO:tensorflow:On image 3900 of 30674
I0209 07:29:31.587508 140107823540096 create_coco_tf_record.py:227] On image 3900 of 30674
INFO:tensorflow:On image 4000 of 30674
I0209 07:29:31.641140 140107823540096 create_coco_tf_record.py:227] On image 4000 of 30674
INFO:tensorflow:On image 4100 of 30674
I0209 07:29:31.706291 140107823540096 create_coco_tf_record.py:227] On image 4100 of 30674
INFO:tensorflow:On image 4200 of 30674
I0209 07:29:31.756948 140107823540096 create_coco_tf_record.py:227] On image 4200 of 30674
INFO:tensorflow:On image 4300 of 30674
I0209 07:29:31.805401 140107823540096 create_coco_tf_record.py:227] On image 4300 of 30674
INFO:tensorflow:On image 4400 of 30674
I0209 07:29:31.855589 140107823540096 create_coco_tf_record.py:227] On image 4400 of 30674
INFO:tensorflow:On image 4500 of 30674
I0209 07:29:31.914511 140107823540096 create_coco_tf_record.py:227] On image 4500 of 30674
INFO:tensorflow:On image 4600 of 30674
I0209 07:29:31.963276 140107823540096 create_coco_tf_record.py:227] On image 4600 of 30674
INFO:tensorflow:On image 4700 of 30674
I0209 07:29:32.013589 140107823540096 create_coco_tf_record.py:227] On image 4700 of 30674
INFO:tensorflow:On image 4800 of 30674
I0209 07:29:32.061713 140107823540096 create_coco_tf_record.py:227] On image 4800 of 30674
INFO:tensorflow:On image 4900 of 30674
I0209 07:29:32.108964 140107823540096 create_coco_tf_record.py:227] On image 4900 of 30674
INFO:tensorflow:On image 5000 of 30674
I0209 07:29:32.158765 140107823540096 create_coco_tf_record.py:227] On image 5000 of 30674
INFO:tensorflow:On image 5100 of 30674
I0209 07:29:32.206059 140107823540096 create_coco_tf_record.py:227] On image 5100 of 30674
INFO:tensorflow:On image 5200 of 30674
I0209 07:29:32.253255 140107823540096 create_coco_tf_record.py:227] On image 5200 of 30674
INFO:tensorflow:On image 5300 of 30674
I0209 07:29:32.301648 140107823540096 create_coco_tf_record.py:227] On image 5300 of 30674
INFO:tensorflow:On image 5400 of 30674
I0209 07:29:32.349104 140107823540096 create_coco_tf_record.py:227] On image 5400 of 30674
INFO:tensorflow:On image 5500 of 30674
I0209 07:29:32.397424 140107823540096 create_coco_tf_record.py:227] On image 5500 of 30674
INFO:tensorflow:On image 5600 of 30674
I0209 07:29:32.445731 140107823540096 create_coco_tf_record.py:227] On image 5600 of 30674
INFO:tensorflow:On image 5700 of 30674
I0209 07:29:32.498605 140107823540096 create_coco_tf_record.py:227] On image 5700 of 30674
INFO:tensorflow:On image 5800 of 30674
I0209 07:29:32.550360 140107823540096 create_coco_tf_record.py:227] On image 5800 of 30674
INFO:tensorflow:On image 5900 of 30674
I0209 07:29:32.603603 140107823540096 create_coco_tf_record.py:227] On image 5900 of 30674
INFO:tensorflow:On image 6000 of 30674
I0209 07:29:32.663421 140107823540096 create_coco_tf_record.py:227] On image 6000 of 30674
INFO:tensorflow:On image 6100 of 30674
I0209 07:29:32.715036 140107823540096 create_coco_tf_record.py:227] On image 6100 of 30674
INFO:tensorflow:On image 6200 of 30674
I0209 07:29:32.766527 140107823540096 create_coco_tf_record.py:227] On image 6200 of 30674
INFO:tensorflow:On image 6300 of 30674
I0209 07:29:32.817535 140107823540096 create_coco_tf_record.py:227] On image 6300 of 30674
INFO:tensorflow:On image 6400 of 30674
I0209 07:29:32.868805 140107823540096 create_coco_tf_record.py:227] On image 6400 of 30674
INFO:tensorflow:On image 6500 of 30674
I0209 07:29:32.917942 140107823540096 create_coco_tf_record.py:227] On image 6500 of 30674
INFO:tensorflow:On image 6600 of 30674
I0209 07:29:32.966505 140107823540096 create_coco_tf_record.py:227] On image 6600 of 30674
INFO:tensorflow:On image 6700 of 30674
I0209 07:29:33.017308 140107823540096 create_coco_tf_record.py:227] On image 6700 of 30674
INFO:tensorflow:On image 6800 of 30674
I0209 07:29:33.070329 140107823540096 create_coco_tf_record.py:227] On image 6800 of 30674
INFO:tensorflow:On image 6900 of 30674
I0209 07:29:33.121753 140107823540096 create_coco_tf_record.py:227] On image 6900 of 30674
INFO:tensorflow:On image 7000 of 30674
I0209 07:29:33.172006 140107823540096 create_coco_tf_record.py:227] On image 7000 of 30674
INFO:tensorflow:On image 7100 of 30674
I0209 07:29:33.221570 140107823540096 create_coco_tf_record.py:227] On image 7100 of 30674
INFO:tensorflow:On image 7200 of 30674
I0209 07:29:33.272365 140107823540096 create_coco_tf_record.py:227] On image 7200 of 30674
INFO:tensorflow:On image 7300 of 30674
I0209 07:29:33.323337 140107823540096 create_coco_tf_record.py:227] On image 7300 of 30674
INFO:tensorflow:On image 7400 of 30674
I0209 07:29:33.374330 140107823540096 create_coco_tf_record.py:227] On image 7400 of 30674
INFO:tensorflow:On image 7500 of 30674
I0209 07:29:33.427039 140107823540096 create_coco_tf_record.py:227] On image 7500 of 30674
INFO:tensorflow:On image 7600 of 30674
I0209 07:29:33.480286 140107823540096 create_coco_tf_record.py:227] On image 7600 of 30674
INFO:tensorflow:On image 7700 of 30674
I0209 07:29:33.534773 140107823540096 create_coco_tf_record.py:227] On image 7700 of 30674
INFO:tensorflow:On image 7800 of 30674
I0209 07:29:33.588568 140107823540096 create_coco_tf_record.py:227] On image 7800 of 30674
INFO:tensorflow:On image 7900 of 30674
I0209 07:29:33.641003 140107823540096 create_coco_tf_record.py:227] On image 7900 of 30674
INFO:tensorflow:On image 8000 of 30674
I0209 07:29:33.699128 140107823540096 create_coco_tf_record.py:227] On image 8000 of 30674
INFO:tensorflow:On image 8100 of 30674
I0209 07:29:33.751177 140107823540096 create_coco_tf_record.py:227] On image 8100 of 30674
INFO:tensorflow:On image 8200 of 30674
I0209 07:29:33.804744 140107823540096 create_coco_tf_record.py:227] On image 8200 of 30674
INFO:tensorflow:On image 8300 of 30674
I0209 07:29:33.856283 140107823540096 create_coco_tf_record.py:227] On image 8300 of 30674
INFO:tensorflow:On image 8400 of 30674
I0209 07:29:33.907517 140107823540096 create_coco_tf_record.py:227] On image 8400 of 30674
INFO:tensorflow:On image 8500 of 30674
I0209 07:29:33.957587 140107823540096 create_coco_tf_record.py:227] On image 8500 of 30674
INFO:tensorflow:On image 8600 of 30674
I0209 07:29:34.009991 140107823540096 create_coco_tf_record.py:227] On image 8600 of 30674
INFO:tensorflow:On image 8700 of 30674
I0209 07:29:34.062829 140107823540096 create_coco_tf_record.py:227] On image 8700 of 30674
INFO:tensorflow:On image 8800 of 30674
I0209 07:29:34.113940 140107823540096 create_coco_tf_record.py:227] On image 8800 of 30674
INFO:tensorflow:On image 8900 of 30674
I0209 07:29:34.162796 140107823540096 create_coco_tf_record.py:227] On image 8900 of 30674
INFO:tensorflow:On image 9000 of 30674
I0209 07:29:34.212890 140107823540096 create_coco_tf_record.py:227] On image 9000 of 30674
INFO:tensorflow:On image 9100 of 30674
I0209 07:29:34.262891 140107823540096 create_coco_tf_record.py:227] On image 9100 of 30674
INFO:tensorflow:On image 9200 of 30674
I0209 07:29:34.312553 140107823540096 create_coco_tf_record.py:227] On image 9200 of 30674
INFO:tensorflow:On image 9300 of 30674
I0209 07:29:34.360566 140107823540096 create_coco_tf_record.py:227] On image 9300 of 30674
INFO:tensorflow:On image 9400 of 30674
I0209 07:29:34.410058 140107823540096 create_coco_tf_record.py:227] On image 9400 of 30674
INFO:tensorflow:On image 9500 of 30674
I0209 07:29:34.457823 140107823540096 create_coco_tf_record.py:227] On image 9500 of 30674
INFO:tensorflow:On image 9600 of 30674
I0209 07:29:34.506357 140107823540096 create_coco_tf_record.py:227] On image 9600 of 30674
INFO:tensorflow:On image 9700 of 30674
I0209 07:29:34.555117 140107823540096 create_coco_tf_record.py:227] On image 9700 of 30674
INFO:tensorflow:On image 9800 of 30674
I0209 07:29:34.604174 140107823540096 create_coco_tf_record.py:227] On image 9800 of 30674
INFO:tensorflow:On image 9900 of 30674
I0209 07:29:34.652172 140107823540096 create_coco_tf_record.py:227] On image 9900 of 30674
INFO:tensorflow:On image 10000 of 30674
I0209 07:29:34.708081 140107823540096 create_coco_tf_record.py:227] On image 10000 of 30674
INFO:tensorflow:On image 10100 of 30674
I0209 07:29:34.758643 140107823540096 create_coco_tf_record.py:227] On image 10100 of 30674
INFO:tensorflow:On image 10200 of 30674
I0209 07:29:34.811463 140107823540096 create_coco_tf_record.py:227] On image 10200 of 30674
INFO:tensorflow:On image 10300 of 30674
I0209 07:29:34.865577 140107823540096 create_coco_tf_record.py:227] On image 10300 of 30674
INFO:tensorflow:On image 10400 of 30674
I0209 07:29:34.920887 140107823540096 create_coco_tf_record.py:227] On image 10400 of 30674
INFO:tensorflow:On image 10500 of 30674
I0209 07:29:34.968788 140107823540096 create_coco_tf_record.py:227] On image 10500 of 30674
INFO:tensorflow:On image 10600 of 30674
I0209 07:29:35.017005 140107823540096 create_coco_tf_record.py:227] On image 10600 of 30674
INFO:tensorflow:On image 10700 of 30674
I0209 07:29:35.065187 140107823540096 create_coco_tf_record.py:227] On image 10700 of 30674
INFO:tensorflow:On image 10800 of 30674
I0209 07:29:35.112729 140107823540096 create_coco_tf_record.py:227] On image 10800 of 30674
INFO:tensorflow:On image 10900 of 30674
I0209 07:29:35.160662 140107823540096 create_coco_tf_record.py:227] On image 10900 of 30674
INFO:tensorflow:On image 11000 of 30674
I0209 07:29:35.210297 140107823540096 create_coco_tf_record.py:227] On image 11000 of 30674
INFO:tensorflow:On image 11100 of 30674
I0209 07:29:35.259209 140107823540096 create_coco_tf_record.py:227] On image 11100 of 30674
INFO:tensorflow:On image 11200 of 30674
I0209 07:29:35.308124 140107823540096 create_coco_tf_record.py:227] On image 11200 of 30674
INFO:tensorflow:On image 11300 of 30674
I0209 07:29:35.356585 140107823540096 create_coco_tf_record.py:227] On image 11300 of 30674
INFO:tensorflow:On image 11400 of 30674
I0209 07:29:35.406039 140107823540096 create_coco_tf_record.py:227] On image 11400 of 30674
INFO:tensorflow:On image 11500 of 30674
I0209 07:29:35.453847 140107823540096 create_coco_tf_record.py:227] On image 11500 of 30674
INFO:tensorflow:On image 11600 of 30674
I0209 07:29:35.501962 140107823540096 create_coco_tf_record.py:227] On image 11600 of 30674
INFO:tensorflow:On image 11700 of 30674
I0209 07:29:35.550579 140107823540096 create_coco_tf_record.py:227] On image 11700 of 30674
INFO:tensorflow:On image 11800 of 30674
I0209 07:29:35.600432 140107823540096 create_coco_tf_record.py:227] On image 11800 of 30674
INFO:tensorflow:On image 11900 of 30674
I0209 07:29:35.648861 140107823540096 create_coco_tf_record.py:227] On image 11900 of 30674
INFO:tensorflow:On image 12000 of 30674
I0209 07:29:35.703112 140107823540096 create_coco_tf_record.py:227] On image 12000 of 30674
INFO:tensorflow:On image 12100 of 30674
I0209 07:29:35.751890 140107823540096 create_coco_tf_record.py:227] On image 12100 of 30674
INFO:tensorflow:On image 12200 of 30674
I0209 07:29:35.800720 140107823540096 create_coco_tf_record.py:227] On image 12200 of 30674
INFO:tensorflow:On image 12300 of 30674
I0209 07:29:35.849707 140107823540096 create_coco_tf_record.py:227] On image 12300 of 30674
INFO:tensorflow:On image 12400 of 30674
I0209 07:29:35.897886 140107823540096 create_coco_tf_record.py:227] On image 12400 of 30674
INFO:tensorflow:On image 12500 of 30674
I0209 07:29:35.946547 140107823540096 create_coco_tf_record.py:227] On image 12500 of 30674
INFO:tensorflow:On image 12600 of 30674
I0209 07:29:35.994341 140107823540096 create_coco_tf_record.py:227] On image 12600 of 30674
INFO:tensorflow:On image 12700 of 30674
I0209 07:29:36.041957 140107823540096 create_coco_tf_record.py:227] On image 12700 of 30674
INFO:tensorflow:On image 12800 of 30674
I0209 07:29:36.090595 140107823540096 create_coco_tf_record.py:227] On image 12800 of 30674
INFO:tensorflow:On image 12900 of 30674
I0209 07:29:36.137639 140107823540096 create_coco_tf_record.py:227] On image 12900 of 30674
INFO:tensorflow:On image 13000 of 30674
I0209 07:29:36.184721 140107823540096 create_coco_tf_record.py:227] On image 13000 of 30674
INFO:tensorflow:On image 13100 of 30674
I0209 07:29:36.231119 140107823540096 create_coco_tf_record.py:227] On image 13100 of 30674
INFO:tensorflow:On image 13200 of 30674
I0209 07:29:36.279808 140107823540096 create_coco_tf_record.py:227] On image 13200 of 30674
INFO:tensorflow:On image 13300 of 30674
I0209 07:29:36.326981 140107823540096 create_coco_tf_record.py:227] On image 13300 of 30674
INFO:tensorflow:On image 13400 of 30674
I0209 07:29:36.391495 140107823540096 create_coco_tf_record.py:227] On image 13400 of 30674
INFO:tensorflow:On image 13500 of 30674
I0209 07:29:36.439026 140107823540096 create_coco_tf_record.py:227] On image 13500 of 30674
INFO:tensorflow:On image 13600 of 30674
I0209 07:29:36.487044 140107823540096 create_coco_tf_record.py:227] On image 13600 of 30674
INFO:tensorflow:On image 13700 of 30674
I0209 07:29:36.534248 140107823540096 create_coco_tf_record.py:227] On image 13700 of 30674
INFO:tensorflow:On image 13800 of 30674
I0209 07:29:36.582198 140107823540096 create_coco_tf_record.py:227] On image 13800 of 30674
INFO:tensorflow:On image 13900 of 30674
I0209 07:29:36.630027 140107823540096 create_coco_tf_record.py:227] On image 13900 of 30674
INFO:tensorflow:On image 14000 of 30674
I0209 07:29:36.697473 140107823540096 create_coco_tf_record.py:227] On image 14000 of 30674
INFO:tensorflow:On image 14100 of 30674
I0209 07:29:36.750743 140107823540096 create_coco_tf_record.py:227] On image 14100 of 30674
INFO:tensorflow:On image 14200 of 30674
I0209 07:29:36.798691 140107823540096 create_coco_tf_record.py:227] On image 14200 of 30674
INFO:tensorflow:On image 14300 of 30674
I0209 07:29:36.847404 140107823540096 create_coco_tf_record.py:227] On image 14300 of 30674
INFO:tensorflow:On image 14400 of 30674
I0209 07:29:36.900060 140107823540096 create_coco_tf_record.py:227] On image 14400 of 30674
INFO:tensorflow:On image 14500 of 30674
I0209 07:29:36.957095 140107823540096 create_coco_tf_record.py:227] On image 14500 of 30674
INFO:tensorflow:On image 14600 of 30674
I0209 07:29:37.009072 140107823540096 create_coco_tf_record.py:227] On image 14600 of 30674
INFO:tensorflow:On image 14700 of 30674
I0209 07:29:37.059747 140107823540096 create_coco_tf_record.py:227] On image 14700 of 30674
INFO:tensorflow:On image 14800 of 30674
I0209 07:29:37.109611 140107823540096 create_coco_tf_record.py:227] On image 14800 of 30674
INFO:tensorflow:On image 14900 of 30674
I0209 07:29:37.163706 140107823540096 create_coco_tf_record.py:227] On image 14900 of 30674
INFO:tensorflow:On image 15000 of 30674
I0209 07:29:37.215618 140107823540096 create_coco_tf_record.py:227] On image 15000 of 30674
INFO:tensorflow:On image 15100 of 30674
I0209 07:29:37.264515 140107823540096 create_coco_tf_record.py:227] On image 15100 of 30674
INFO:tensorflow:On image 15200 of 30674
I0209 07:29:37.313587 140107823540096 create_coco_tf_record.py:227] On image 15200 of 30674
INFO:tensorflow:On image 15300 of 30674
I0209 07:29:37.366154 140107823540096 create_coco_tf_record.py:227] On image 15300 of 30674
INFO:tensorflow:On image 15400 of 30674
I0209 07:29:37.416051 140107823540096 create_coco_tf_record.py:227] On image 15400 of 30674
INFO:tensorflow:On image 15500 of 30674
I0209 07:29:37.464270 140107823540096 create_coco_tf_record.py:227] On image 15500 of 30674
INFO:tensorflow:On image 15600 of 30674
I0209 07:29:37.513511 140107823540096 create_coco_tf_record.py:227] On image 15600 of 30674
INFO:tensorflow:On image 15700 of 30674
I0209 07:29:37.562650 140107823540096 create_coco_tf_record.py:227] On image 15700 of 30674
INFO:tensorflow:On image 15800 of 30674
I0209 07:29:37.611023 140107823540096 create_coco_tf_record.py:227] On image 15800 of 30674
INFO:tensorflow:On image 15900 of 30674
I0209 07:29:37.659338 140107823540096 create_coco_tf_record.py:227] On image 15900 of 30674
INFO:tensorflow:On image 16000 of 30674
I0209 07:29:37.707298 140107823540096 create_coco_tf_record.py:227] On image 16000 of 30674
INFO:tensorflow:On image 16100 of 30674
I0209 07:29:37.761783 140107823540096 create_coco_tf_record.py:227] On image 16100 of 30674
INFO:tensorflow:On image 16200 of 30674
I0209 07:29:37.810244 140107823540096 create_coco_tf_record.py:227] On image 16200 of 30674
INFO:tensorflow:On image 16300 of 30674
I0209 07:29:37.858860 140107823540096 create_coco_tf_record.py:227] On image 16300 of 30674
INFO:tensorflow:On image 16400 of 30674
I0209 07:29:37.906785 140107823540096 create_coco_tf_record.py:227] On image 16400 of 30674
INFO:tensorflow:On image 16500 of 30674
I0209 07:29:37.954585 140107823540096 create_coco_tf_record.py:227] On image 16500 of 30674
INFO:tensorflow:On image 16600 of 30674
I0209 07:29:38.002529 140107823540096 create_coco_tf_record.py:227] On image 16600 of 30674
INFO:tensorflow:On image 16700 of 30674
I0209 07:29:38.051216 140107823540096 create_coco_tf_record.py:227] On image 16700 of 30674
INFO:tensorflow:On image 16800 of 30674
I0209 07:29:38.098484 140107823540096 create_coco_tf_record.py:227] On image 16800 of 30674
INFO:tensorflow:On image 16900 of 30674
I0209 07:29:38.146377 140107823540096 create_coco_tf_record.py:227] On image 16900 of 30674
INFO:tensorflow:On image 17000 of 30674
I0209 07:29:38.196822 140107823540096 create_coco_tf_record.py:227] On image 17000 of 30674
INFO:tensorflow:On image 17100 of 30674
I0209 07:29:38.245764 140107823540096 create_coco_tf_record.py:227] On image 17100 of 30674
INFO:tensorflow:On image 17200 of 30674
I0209 07:29:38.293771 140107823540096 create_coco_tf_record.py:227] On image 17200 of 30674
INFO:tensorflow:On image 17300 of 30674
I0209 07:29:38.341681 140107823540096 create_coco_tf_record.py:227] On image 17300 of 30674
INFO:tensorflow:On image 17400 of 30674
I0209 07:29:38.390666 140107823540096 create_coco_tf_record.py:227] On image 17400 of 30674
INFO:tensorflow:On image 17500 of 30674
I0209 07:29:38.438493 140107823540096 create_coco_tf_record.py:227] On image 17500 of 30674
INFO:tensorflow:On image 17600 of 30674
I0209 07:29:38.489065 140107823540096 create_coco_tf_record.py:227] On image 17600 of 30674
INFO:tensorflow:On image 17700 of 30674
I0209 07:29:38.545770 140107823540096 create_coco_tf_record.py:227] On image 17700 of 30674
INFO:tensorflow:On image 17800 of 30674
I0209 07:29:38.600012 140107823540096 create_coco_tf_record.py:227] On image 17800 of 30674
INFO:tensorflow:On image 17900 of 30674
I0209 07:29:38.651055 140107823540096 create_coco_tf_record.py:227] On image 17900 of 30674
INFO:tensorflow:On image 18000 of 30674
I0209 07:29:38.700518 140107823540096 create_coco_tf_record.py:227] On image 18000 of 30674
INFO:tensorflow:On image 18100 of 30674
I0209 07:29:38.752946 140107823540096 create_coco_tf_record.py:227] On image 18100 of 30674
INFO:tensorflow:On image 18200 of 30674
I0209 07:29:38.803449 140107823540096 create_coco_tf_record.py:227] On image 18200 of 30674
INFO:tensorflow:On image 18300 of 30674
I0209 07:29:38.852744 140107823540096 create_coco_tf_record.py:227] On image 18300 of 30674
INFO:tensorflow:On image 18400 of 30674
I0209 07:29:38.901023 140107823540096 create_coco_tf_record.py:227] On image 18400 of 30674
INFO:tensorflow:On image 18500 of 30674
I0209 07:29:38.949615 140107823540096 create_coco_tf_record.py:227] On image 18500 of 30674
INFO:tensorflow:On image 18600 of 30674
I0209 07:29:38.997643 140107823540096 create_coco_tf_record.py:227] On image 18600 of 30674
INFO:tensorflow:On image 18700 of 30674
I0209 07:29:39.046510 140107823540096 create_coco_tf_record.py:227] On image 18700 of 30674
INFO:tensorflow:On image 18800 of 30674
I0209 07:29:39.095378 140107823540096 create_coco_tf_record.py:227] On image 18800 of 30674
INFO:tensorflow:On image 18900 of 30674
I0209 07:29:39.143252 140107823540096 create_coco_tf_record.py:227] On image 18900 of 30674
INFO:tensorflow:On image 19000 of 30674
I0209 07:29:39.191761 140107823540096 create_coco_tf_record.py:227] On image 19000 of 30674
INFO:tensorflow:On image 19100 of 30674
I0209 07:29:39.239664 140107823540096 create_coco_tf_record.py:227] On image 19100 of 30674
INFO:tensorflow:On image 19200 of 30674
I0209 07:29:39.288391 140107823540096 create_coco_tf_record.py:227] On image 19200 of 30674
INFO:tensorflow:On image 19300 of 30674
I0209 07:29:39.335295 140107823540096 create_coco_tf_record.py:227] On image 19300 of 30674
INFO:tensorflow:On image 19400 of 30674
I0209 07:29:39.382753 140107823540096 create_coco_tf_record.py:227] On image 19400 of 30674
INFO:tensorflow:On image 19500 of 30674
I0209 07:29:39.430925 140107823540096 create_coco_tf_record.py:227] On image 19500 of 30674
INFO:tensorflow:On image 19600 of 30674
I0209 07:29:39.479413 140107823540096 create_coco_tf_record.py:227] On image 19600 of 30674
INFO:tensorflow:On image 19700 of 30674
I0209 07:29:39.526808 140107823540096 create_coco_tf_record.py:227] On image 19700 of 30674
INFO:tensorflow:On image 19800 of 30674
I0209 07:29:39.578537 140107823540096 create_coco_tf_record.py:227] On image 19800 of 30674
INFO:tensorflow:On image 19900 of 30674
I0209 07:29:39.630142 140107823540096 create_coco_tf_record.py:227] On image 19900 of 30674
INFO:tensorflow:On image 20000 of 30674
I0209 07:29:39.680707 140107823540096 create_coco_tf_record.py:227] On image 20000 of 30674
INFO:tensorflow:On image 20100 of 30674
I0209 07:29:39.731282 140107823540096 create_coco_tf_record.py:227] On image 20100 of 30674
INFO:tensorflow:On image 20200 of 30674
I0209 07:29:39.790047 140107823540096 create_coco_tf_record.py:227] On image 20200 of 30674
INFO:tensorflow:On image 20300 of 30674
I0209 07:29:39.843030 140107823540096 create_coco_tf_record.py:227] On image 20300 of 30674
INFO:tensorflow:On image 20400 of 30674
I0209 07:29:39.892612 140107823540096 create_coco_tf_record.py:227] On image 20400 of 30674
INFO:tensorflow:On image 20500 of 30674
I0209 07:29:39.950912 140107823540096 create_coco_tf_record.py:227] On image 20500 of 30674
INFO:tensorflow:On image 20600 of 30674
I0209 07:29:40.000617 140107823540096 create_coco_tf_record.py:227] On image 20600 of 30674
INFO:tensorflow:On image 20700 of 30674
I0209 07:29:40.050609 140107823540096 create_coco_tf_record.py:227] On image 20700 of 30674
INFO:tensorflow:On image 20800 of 30674
I0209 07:29:40.098548 140107823540096 create_coco_tf_record.py:227] On image 20800 of 30674
INFO:tensorflow:On image 20900 of 30674
I0209 07:29:40.146512 140107823540096 create_coco_tf_record.py:227] On image 20900 of 30674
INFO:tensorflow:On image 21000 of 30674
I0209 07:29:40.195150 140107823540096 create_coco_tf_record.py:227] On image 21000 of 30674
INFO:tensorflow:On image 21100 of 30674
I0209 07:29:40.242985 140107823540096 create_coco_tf_record.py:227] On image 21100 of 30674
INFO:tensorflow:On image 21200 of 30674
I0209 07:29:40.291406 140107823540096 create_coco_tf_record.py:227] On image 21200 of 30674
INFO:tensorflow:On image 21300 of 30674
I0209 07:29:40.338825 140107823540096 create_coco_tf_record.py:227] On image 21300 of 30674
INFO:tensorflow:On image 21400 of 30674
I0209 07:29:40.386161 140107823540096 create_coco_tf_record.py:227] On image 21400 of 30674
INFO:tensorflow:On image 21500 of 30674
I0209 07:29:40.435175 140107823540096 create_coco_tf_record.py:227] On image 21500 of 30674
INFO:tensorflow:On image 21600 of 30674
I0209 07:29:40.483256 140107823540096 create_coco_tf_record.py:227] On image 21600 of 30674
INFO:tensorflow:On image 21700 of 30674
I0209 07:29:40.530771 140107823540096 create_coco_tf_record.py:227] On image 21700 of 30674
INFO:tensorflow:On image 21800 of 30674
I0209 07:29:40.578068 140107823540096 create_coco_tf_record.py:227] On image 21800 of 30674
INFO:tensorflow:On image 21900 of 30674
I0209 07:29:40.625772 140107823540096 create_coco_tf_record.py:227] On image 21900 of 30674
INFO:tensorflow:On image 22000 of 30674
I0209 07:29:40.672758 140107823540096 create_coco_tf_record.py:227] On image 22000 of 30674
INFO:tensorflow:On image 22100 of 30674
I0209 07:29:40.720202 140107823540096 create_coco_tf_record.py:227] On image 22100 of 30674
INFO:tensorflow:On image 22200 of 30674
I0209 07:29:40.768235 140107823540096 create_coco_tf_record.py:227] On image 22200 of 30674
INFO:tensorflow:On image 22300 of 30674
I0209 07:29:40.822160 140107823540096 create_coco_tf_record.py:227] On image 22300 of 30674
INFO:tensorflow:On image 22400 of 30674
I0209 07:29:40.870104 140107823540096 create_coco_tf_record.py:227] On image 22400 of 30674
INFO:tensorflow:On image 22500 of 30674
I0209 07:29:40.917841 140107823540096 create_coco_tf_record.py:227] On image 22500 of 30674
INFO:tensorflow:On image 22600 of 30674
I0209 07:29:40.965408 140107823540096 create_coco_tf_record.py:227] On image 22600 of 30674
INFO:tensorflow:On image 22700 of 30674
I0209 07:29:41.013479 140107823540096 create_coco_tf_record.py:227] On image 22700 of 30674
INFO:tensorflow:On image 22800 of 30674
I0209 07:29:41.061653 140107823540096 create_coco_tf_record.py:227] On image 22800 of 30674
INFO:tensorflow:On image 22900 of 30674
I0209 07:29:41.110107 140107823540096 create_coco_tf_record.py:227] On image 22900 of 30674
INFO:tensorflow:On image 23000 of 30674
I0209 07:29:41.157727 140107823540096 create_coco_tf_record.py:227] On image 23000 of 30674
INFO:tensorflow:On image 23100 of 30674
I0209 07:29:41.207663 140107823540096 create_coco_tf_record.py:227] On image 23100 of 30674
INFO:tensorflow:On image 23200 of 30674
I0209 07:29:41.254960 140107823540096 create_coco_tf_record.py:227] On image 23200 of 30674
INFO:tensorflow:On image 23300 of 30674
I0209 07:29:41.304700 140107823540096 create_coco_tf_record.py:227] On image 23300 of 30674
INFO:tensorflow:On image 23400 of 30674
I0209 07:29:41.352308 140107823540096 create_coco_tf_record.py:227] On image 23400 of 30674
INFO:tensorflow:On image 23500 of 30674
I0209 07:29:41.401085 140107823540096 create_coco_tf_record.py:227] On image 23500 of 30674
INFO:tensorflow:On image 23600 of 30674
I0209 07:29:41.449143 140107823540096 create_coco_tf_record.py:227] On image 23600 of 30674
INFO:tensorflow:On image 23700 of 30674
I0209 07:29:41.496366 140107823540096 create_coco_tf_record.py:227] On image 23700 of 30674
INFO:tensorflow:On image 23800 of 30674
I0209 07:29:41.545144 140107823540096 create_coco_tf_record.py:227] On image 23800 of 30674
INFO:tensorflow:On image 23900 of 30674
I0209 07:29:41.611638 140107823540096 create_coco_tf_record.py:227] On image 23900 of 30674
INFO:tensorflow:On image 24000 of 30674
I0209 07:29:41.665069 140107823540096 create_coco_tf_record.py:227] On image 24000 of 30674
INFO:tensorflow:On image 24100 of 30674
I0209 07:29:41.717655 140107823540096 create_coco_tf_record.py:227] On image 24100 of 30674
INFO:tensorflow:On image 24200 of 30674
I0209 07:29:41.776913 140107823540096 create_coco_tf_record.py:227] On image 24200 of 30674
INFO:tensorflow:On image 24300 of 30674
I0209 07:29:41.858232 140107823540096 create_coco_tf_record.py:227] On image 24300 of 30674
INFO:tensorflow:On image 24400 of 30674
I0209 07:29:41.921696 140107823540096 create_coco_tf_record.py:227] On image 24400 of 30674
INFO:tensorflow:On image 24500 of 30674
I0209 07:29:41.979680 140107823540096 create_coco_tf_record.py:227] On image 24500 of 30674
INFO:tensorflow:On image 24600 of 30674
I0209 07:29:42.035171 140107823540096 create_coco_tf_record.py:227] On image 24600 of 30674
INFO:tensorflow:On image 24700 of 30674
I0209 07:29:42.086365 140107823540096 create_coco_tf_record.py:227] On image 24700 of 30674
INFO:tensorflow:On image 24800 of 30674
I0209 07:29:42.137154 140107823540096 create_coco_tf_record.py:227] On image 24800 of 30674
INFO:tensorflow:On image 24900 of 30674
I0209 07:29:42.187874 140107823540096 create_coco_tf_record.py:227] On image 24900 of 30674
INFO:tensorflow:On image 25000 of 30674
I0209 07:29:42.239138 140107823540096 create_coco_tf_record.py:227] On image 25000 of 30674
INFO:tensorflow:On image 25100 of 30674
I0209 07:29:42.289384 140107823540096 create_coco_tf_record.py:227] On image 25100 of 30674
INFO:tensorflow:On image 25200 of 30674
I0209 07:29:42.340937 140107823540096 create_coco_tf_record.py:227] On image 25200 of 30674
INFO:tensorflow:On image 25300 of 30674
I0209 07:29:42.392171 140107823540096 create_coco_tf_record.py:227] On image 25300 of 30674
INFO:tensorflow:On image 25400 of 30674
I0209 07:29:42.442649 140107823540096 create_coco_tf_record.py:227] On image 25400 of 30674
INFO:tensorflow:On image 25500 of 30674
I0209 07:29:42.493071 140107823540096 create_coco_tf_record.py:227] On image 25500 of 30674
INFO:tensorflow:On image 25600 of 30674
I0209 07:29:42.542949 140107823540096 create_coco_tf_record.py:227] On image 25600 of 30674
INFO:tensorflow:On image 25700 of 30674
I0209 07:29:42.591792 140107823540096 create_coco_tf_record.py:227] On image 25700 of 30674
INFO:tensorflow:On image 25800 of 30674
I0209 07:29:42.640826 140107823540096 create_coco_tf_record.py:227] On image 25800 of 30674
INFO:tensorflow:On image 25900 of 30674
I0209 07:29:42.689953 140107823540096 create_coco_tf_record.py:227] On image 25900 of 30674
INFO:tensorflow:On image 26000 of 30674
I0209 07:29:42.738588 140107823540096 create_coco_tf_record.py:227] On image 26000 of 30674
INFO:tensorflow:On image 26100 of 30674
I0209 07:29:42.787057 140107823540096 create_coco_tf_record.py:227] On image 26100 of 30674
INFO:tensorflow:On image 26200 of 30674
I0209 07:29:42.836802 140107823540096 create_coco_tf_record.py:227] On image 26200 of 30674
INFO:tensorflow:On image 26300 of 30674
I0209 07:29:42.891639 140107823540096 create_coco_tf_record.py:227] On image 26300 of 30674
INFO:tensorflow:On image 26400 of 30674
I0209 07:29:42.939060 140107823540096 create_coco_tf_record.py:227] On image 26400 of 30674
INFO:tensorflow:On image 26500 of 30674
I0209 07:29:42.987085 140107823540096 create_coco_tf_record.py:227] On image 26500 of 30674
INFO:tensorflow:On image 26600 of 30674
I0209 07:29:43.038125 140107823540096 create_coco_tf_record.py:227] On image 26600 of 30674
INFO:tensorflow:On image 26700 of 30674
I0209 07:29:43.086430 140107823540096 create_coco_tf_record.py:227] On image 26700 of 30674
INFO:tensorflow:On image 26800 of 30674
I0209 07:29:43.135753 140107823540096 create_coco_tf_record.py:227] On image 26800 of 30674
INFO:tensorflow:On image 26900 of 30674
I0209 07:29:43.183268 140107823540096 create_coco_tf_record.py:227] On image 26900 of 30674
INFO:tensorflow:On image 27000 of 30674
I0209 07:29:43.231203 140107823540096 create_coco_tf_record.py:227] On image 27000 of 30674
INFO:tensorflow:On image 27100 of 30674
I0209 07:29:43.279075 140107823540096 create_coco_tf_record.py:227] On image 27100 of 30674
INFO:tensorflow:On image 27200 of 30674
I0209 07:29:43.328093 140107823540096 create_coco_tf_record.py:227] On image 27200 of 30674
INFO:tensorflow:On image 27300 of 30674
I0209 07:29:43.375055 140107823540096 create_coco_tf_record.py:227] On image 27300 of 30674
INFO:tensorflow:On image 27400 of 30674
I0209 07:29:43.424542 140107823540096 create_coco_tf_record.py:227] On image 27400 of 30674
INFO:tensorflow:On image 27500 of 30674
I0209 07:29:43.473034 140107823540096 create_coco_tf_record.py:227] On image 27500 of 30674
INFO:tensorflow:On image 27600 of 30674
I0209 07:29:43.520908 140107823540096 create_coco_tf_record.py:227] On image 27600 of 30674
INFO:tensorflow:On image 27700 of 30674
I0209 07:29:43.569979 140107823540096 create_coco_tf_record.py:227] On image 27700 of 30674
INFO:tensorflow:On image 27800 of 30674
I0209 07:29:43.623033 140107823540096 create_coco_tf_record.py:227] On image 27800 of 30674
INFO:tensorflow:On image 27900 of 30674
I0209 07:29:43.676839 140107823540096 create_coco_tf_record.py:227] On image 27900 of 30674
INFO:tensorflow:On image 28000 of 30674
I0209 07:29:43.729416 140107823540096 create_coco_tf_record.py:227] On image 28000 of 30674
INFO:tensorflow:On image 28100 of 30674
I0209 07:29:43.783208 140107823540096 create_coco_tf_record.py:227] On image 28100 of 30674
INFO:tensorflow:On image 28200 of 30674
I0209 07:29:43.837840 140107823540096 create_coco_tf_record.py:227] On image 28200 of 30674
INFO:tensorflow:On image 28300 of 30674
I0209 07:29:43.896660 140107823540096 create_coco_tf_record.py:227] On image 28300 of 30674
INFO:tensorflow:On image 28400 of 30674
I0209 07:29:43.946526 140107823540096 create_coco_tf_record.py:227] On image 28400 of 30674
INFO:tensorflow:On image 28500 of 30674
I0209 07:29:43.994785 140107823540096 create_coco_tf_record.py:227] On image 28500 of 30674
INFO:tensorflow:On image 28600 of 30674
I0209 07:29:44.043161 140107823540096 create_coco_tf_record.py:227] On image 28600 of 30674
INFO:tensorflow:On image 28700 of 30674
I0209 07:29:44.092373 140107823540096 create_coco_tf_record.py:227] On image 28700 of 30674
INFO:tensorflow:On image 28800 of 30674
I0209 07:29:44.143258 140107823540096 create_coco_tf_record.py:227] On image 28800 of 30674
INFO:tensorflow:On image 28900 of 30674
I0209 07:29:44.194888 140107823540096 create_coco_tf_record.py:227] On image 28900 of 30674
INFO:tensorflow:On image 29000 of 30674
I0209 07:29:44.247044 140107823540096 create_coco_tf_record.py:227] On image 29000 of 30674
INFO:tensorflow:On image 29100 of 30674
I0209 07:29:44.299930 140107823540096 create_coco_tf_record.py:227] On image 29100 of 30674
INFO:tensorflow:On image 29200 of 30674
I0209 07:29:44.353611 140107823540096 create_coco_tf_record.py:227] On image 29200 of 30674
INFO:tensorflow:On image 29300 of 30674
I0209 07:29:44.405689 140107823540096 create_coco_tf_record.py:227] On image 29300 of 30674
INFO:tensorflow:On image 29400 of 30674
I0209 07:29:44.454727 140107823540096 create_coco_tf_record.py:227] On image 29400 of 30674
INFO:tensorflow:On image 29500 of 30674
I0209 07:29:44.502712 140107823540096 create_coco_tf_record.py:227] On image 29500 of 30674
INFO:tensorflow:On image 29600 of 30674
I0209 07:29:44.551148 140107823540096 create_coco_tf_record.py:227] On image 29600 of 30674
INFO:tensorflow:On image 29700 of 30674
I0209 07:29:44.599494 140107823540096 create_coco_tf_record.py:227] On image 29700 of 30674
INFO:tensorflow:On image 29800 of 30674
I0209 07:29:44.646869 140107823540096 create_coco_tf_record.py:227] On image 29800 of 30674
INFO:tensorflow:On image 29900 of 30674
I0209 07:29:44.694324 140107823540096 create_coco_tf_record.py:227] On image 29900 of 30674
INFO:tensorflow:On image 30000 of 30674
I0209 07:29:44.741466 140107823540096 create_coco_tf_record.py:227] On image 30000 of 30674
INFO:tensorflow:On image 30100 of 30674
I0209 07:29:44.788800 140107823540096 create_coco_tf_record.py:227] On image 30100 of 30674
INFO:tensorflow:On image 30200 of 30674
I0209 07:29:44.836947 140107823540096 create_coco_tf_record.py:227] On image 30200 of 30674
INFO:tensorflow:On image 30300 of 30674
I0209 07:29:44.890257 140107823540096 create_coco_tf_record.py:227] On image 30300 of 30674
INFO:tensorflow:On image 30400 of 30674
I0209 07:29:44.940052 140107823540096 create_coco_tf_record.py:227] On image 30400 of 30674
INFO:tensorflow:On image 30500 of 30674
I0209 07:29:44.987952 140107823540096 create_coco_tf_record.py:227] On image 30500 of 30674
INFO:tensorflow:On image 30600 of 30674
I0209 07:29:45.036247 140107823540096 create_coco_tf_record.py:227] On image 30600 of 30674
INFO:tensorflow:Finished writing, skipped 652 annotations.
I0209 07:29:45.072056 140107823540096 create_coco_tf_record.py:234] Finished writing, skipped 652 annotations.
INFO:tensorflow:Found groundtruth annotations. Building annotations index.
I0209 07:29:45.282140 140107823540096 create_coco_tf_record.py:209] Found groundtruth annotations. Building annotations index.
INFO:tensorflow:0 images are missing annotations.
I0209 07:29:45.285487 140107823540096 create_coco_tf_record.py:222] 0 images are missing annotations.
INFO:tensorflow:On image 0 of 4381
I0209 07:29:45.285650 140107823540096 create_coco_tf_record.py:227] On image 0 of 4381
INFO:tensorflow:On image 100 of 4381
I0209 07:29:45.338083 140107823540096 create_coco_tf_record.py:227] On image 100 of 4381
INFO:tensorflow:On image 200 of 4381
I0209 07:29:45.391104 140107823540096 create_coco_tf_record.py:227] On image 200 of 4381
INFO:tensorflow:On image 300 of 4381
I0209 07:29:45.441899 140107823540096 create_coco_tf_record.py:227] On image 300 of 4381
INFO:tensorflow:On image 400 of 4381
I0209 07:29:45.492035 140107823540096 create_coco_tf_record.py:227] On image 400 of 4381
INFO:tensorflow:On image 500 of 4381
I0209 07:29:45.542772 140107823540096 create_coco_tf_record.py:227] On image 500 of 4381
INFO:tensorflow:On image 600 of 4381
I0209 07:29:45.593809 140107823540096 create_coco_tf_record.py:227] On image 600 of 4381
INFO:tensorflow:On image 700 of 4381
I0209 07:29:45.646643 140107823540096 create_coco_tf_record.py:227] On image 700 of 4381
INFO:tensorflow:On image 800 of 4381
I0209 07:29:45.700170 140107823540096 create_coco_tf_record.py:227] On image 800 of 4381
INFO:tensorflow:On image 900 of 4381
I0209 07:29:45.753659 140107823540096 create_coco_tf_record.py:227] On image 900 of 4381
INFO:tensorflow:On image 1000 of 4381
I0209 07:29:45.806349 140107823540096 create_coco_tf_record.py:227] On image 1000 of 4381
INFO:tensorflow:On image 1100 of 4381
I0209 07:29:45.859146 140107823540096 create_coco_tf_record.py:227] On image 1100 of 4381
INFO:tensorflow:On image 1200 of 4381
I0209 07:29:45.915701 140107823540096 create_coco_tf_record.py:227] On image 1200 of 4381
INFO:tensorflow:On image 1300 of 4381
I0209 07:29:45.964930 140107823540096 create_coco_tf_record.py:227] On image 1300 of 4381
INFO:tensorflow:On image 1400 of 4381
I0209 07:29:46.013833 140107823540096 create_coco_tf_record.py:227] On image 1400 of 4381
INFO:tensorflow:On image 1500 of 4381
I0209 07:29:46.064570 140107823540096 create_coco_tf_record.py:227] On image 1500 of 4381
INFO:tensorflow:On image 1600 of 4381
I0209 07:29:46.114058 140107823540096 create_coco_tf_record.py:227] On image 1600 of 4381
INFO:tensorflow:On image 1700 of 4381
I0209 07:29:46.163121 140107823540096 create_coco_tf_record.py:227] On image 1700 of 4381
INFO:tensorflow:On image 1800 of 4381
I0209 07:29:46.211823 140107823540096 create_coco_tf_record.py:227] On image 1800 of 4381
INFO:tensorflow:On image 1900 of 4381
I0209 07:29:46.260781 140107823540096 create_coco_tf_record.py:227] On image 1900 of 4381
INFO:tensorflow:On image 2000 of 4381
I0209 07:29:46.309998 140107823540096 create_coco_tf_record.py:227] On image 2000 of 4381
INFO:tensorflow:On image 2100 of 4381
I0209 07:29:46.358225 140107823540096 create_coco_tf_record.py:227] On image 2100 of 4381
INFO:tensorflow:On image 2200 of 4381
I0209 07:29:46.408882 140107823540096 create_coco_tf_record.py:227] On image 2200 of 4381
INFO:tensorflow:On image 2300 of 4381
I0209 07:29:46.457455 140107823540096 create_coco_tf_record.py:227] On image 2300 of 4381
INFO:tensorflow:On image 2400 of 4381
I0209 07:29:46.505949 140107823540096 create_coco_tf_record.py:227] On image 2400 of 4381
INFO:tensorflow:On image 2500 of 4381
I0209 07:29:46.554764 140107823540096 create_coco_tf_record.py:227] On image 2500 of 4381
INFO:tensorflow:On image 2600 of 4381
I0209 07:29:46.603295 140107823540096 create_coco_tf_record.py:227] On image 2600 of 4381
INFO:tensorflow:On image 2700 of 4381
I0209 07:29:46.656020 140107823540096 create_coco_tf_record.py:227] On image 2700 of 4381
INFO:tensorflow:On image 2800 of 4381
I0209 07:29:46.707798 140107823540096 create_coco_tf_record.py:227] On image 2800 of 4381
INFO:tensorflow:On image 2900 of 4381
I0209 07:29:46.759378 140107823540096 create_coco_tf_record.py:227] On image 2900 of 4381
INFO:tensorflow:On image 3000 of 4381
I0209 07:29:46.810447 140107823540096 create_coco_tf_record.py:227] On image 3000 of 4381
INFO:tensorflow:On image 3100 of 4381
I0209 07:29:46.861093 140107823540096 create_coco_tf_record.py:227] On image 3100 of 4381
INFO:tensorflow:On image 3200 of 4381
I0209 07:29:46.945095 140107823540096 create_coco_tf_record.py:227] On image 3200 of 4381
INFO:tensorflow:On image 3300 of 4381
I0209 07:29:46.999625 140107823540096 create_coco_tf_record.py:227] On image 3300 of 4381
INFO:tensorflow:On image 3400 of 4381
I0209 07:29:47.050895 140107823540096 create_coco_tf_record.py:227] On image 3400 of 4381
INFO:tensorflow:On image 3500 of 4381
I0209 07:29:47.101934 140107823540096 create_coco_tf_record.py:227] On image 3500 of 4381
INFO:tensorflow:On image 3600 of 4381
I0209 07:29:47.156166 140107823540096 create_coco_tf_record.py:227] On image 3600 of 4381
INFO:tensorflow:On image 3700 of 4381
I0209 07:29:47.215608 140107823540096 create_coco_tf_record.py:227] On image 3700 of 4381
INFO:tensorflow:On image 3800 of 4381
I0209 07:29:47.272501 140107823540096 create_coco_tf_record.py:227] On image 3800 of 4381
INFO:tensorflow:On image 3900 of 4381
I0209 07:29:47.326143 140107823540096 create_coco_tf_record.py:227] On image 3900 of 4381
INFO:tensorflow:On image 4000 of 4381
I0209 07:29:47.381117 140107823540096 create_coco_tf_record.py:227] On image 4000 of 4381
INFO:tensorflow:On image 4100 of 4381
I0209 07:29:47.436727 140107823540096 create_coco_tf_record.py:227] On image 4100 of 4381
INFO:tensorflow:On image 4200 of 4381
I0209 07:29:47.487106 140107823540096 create_coco_tf_record.py:227] On image 4200 of 4381
INFO:tensorflow:On image 4300 of 4381
I0209 07:29:47.540182 140107823540096 create_coco_tf_record.py:227] On image 4300 of 4381
INFO:tensorflow:Finished writing, skipped 103 annotations.
I0209 07:29:47.582033 140107823540096 create_coco_tf_record.py:234] Finished writing, skipped 103 annotations.
Load dataset
[ ]
dataset = 'cifar10'
batch_size = 512
config = common_config.with_dataset(common_config.get_config(), dataset)
num_classes = input_pipeline.get_dataset_info(dataset, 'train')['num_classes']
config.batch = batch_size
config.pp.crop = 224
INFO:absl:Load pre-computed DatasetInfo (eg: splits, num examples,...) from GCS: cifar10/3.0.2
INFO:absl:Load dataset info from /tmp/tmp8pyec2amtfds
INFO:absl:Field info.citation from disk and from code do not match. Keeping the one from code.
[ ]
# For details about setting up datasets, see input_pipeline.py on the right.
ds_train = input_pipeline.get_data_from_tfds(config=config, mode='train')
ds_test = input_pipeline.get_data_from_tfds(config=config, mode='test')

del config  # Only needed to instantiate datasets.

[ ]
# Fetch a batch of test images for illustration purposes.
batch = next(iter(ds_test.as_numpy_iterator()))
# Note the shape : [num_local_devices, local_batch_size, h, w, c]
batch['image'].shape
(1, 512, 224, 224, 3)
[ ]
# Show some imags with their labels.
images, labels = batch['image'][0][:9], batch['label'][0][:9]
titles = map(make_label_getter(dataset), labels.argmax(axis=1))
show_img_grid(images, titles)

[ ]
# Same as above, but with train images.
# Note how images are cropped/scaled differently.
# Check out input_pipeline.get_data() in the editor at your right to see how the
# images are preprocessed differently.
batch = next(iter(ds_train.as_numpy_iterator()))
images, labels = batch['image'][0][:9], batch['label'][0][:9]
titles = map(make_label_getter(dataset), labels.argmax(axis=1))
show_img_grid(images, titles)

Load pre-trained
[ ]
model_config = models_config.MODEL_CONFIGS[model_name]
model_config
classifier: token
hidden_size: 768
name: ViT-B_32
patches:
  size: !!python/tuple
  - 32
  - 32
representation_size: null
transformer:
  attention_dropout_rate: 0.0
  dropout_rate: 0.0
  mlp_dim: 3072
  num_heads: 12
  num_layers: 12
[ ]
# Load model definition & initialize random parameters.
# This also compiles the model to XLA (takes some minutes the first time).
if model_name.startswith('Mixer'):
  model = models.MlpMixer(num_classes=num_classes, **model_config)
else:
  model = models.VisionTransformer(num_classes=num_classes, **model_config)
variables = jax.jit(lambda: model.init(
    jax.random.PRNGKey(0),
    # Discard the "num_local_devices" dimension of the batch for initialization.
    batch['image'][0, :1],
    train=False,
), backend='cpu')()
[ ]
# Load and convert pretrained checkpoint.
# This involves loading the actual pre-trained model results, but then also also
# modifying the parameters a bit, e.g. changing the final layers, and resizing
# the positional embeddings.
# For details, refer to the code and to the methods of the paper.
params = checkpoint.load_pretrained(
    pretrained_path=f'{model_name}.npz',
    init_params=variables['params'],
    model_config=model_config,
)
INFO:absl:Inspect extra keys:
{'pre_logits/bias', 'pre_logits/kernel'}
INFO:absl:load_pretrained: drop-head variant
Evaluate
[ ]
# So far, all our data is in the host memory. Let's now replicate the arrays
# into the devices.
# This will make every array in the pytree params become a ShardedDeviceArray
# that has the same data replicated across all local devices.
# For TPU it replicates the params in every core.
# For a single GPU this simply moves the data onto the device.
# For CPU it simply creates a copy.
params_repl = flax.jax_utils.replicate(params)
print('params.cls:', type(params['head']['bias']).__name__,
      params['head']['bias'].shape)
print('params_repl.cls:', type(params_repl['head']['bias']).__name__,
      params_repl['head']['bias'].shape)
params.cls: DeviceArray (10,)
params_repl.cls: ShardedDeviceArray (1, 10)
/usr/local/lib/python3.7/dist-packages/jax/lib/xla_bridge.py:317: UserWarning: jax.host_count has been renamed to jax.process_count. This alias will eventually be removed; please update your code.
  "jax.host_count has been renamed to jax.process_count. This alias "
/usr/local/lib/python3.7/dist-packages/jax/lib/xla_bridge.py:304: UserWarning: jax.host_id has been renamed to jax.process_index. This alias will eventually be removed; please update your code.
  "jax.host_id has been renamed to jax.process_index. This alias "
[ ]
# Then map the call to our model's forward pass onto all available devices.
vit_apply_repl = jax.pmap(lambda params, inputs: model.apply(
    dict(params=params), inputs, train=False))
[ ]
def get_accuracy(params_repl):
  """Returns accuracy evaluated on the test set."""
  good = total = 0
  steps = input_pipeline.get_dataset_info(dataset, 'test')['num_examples'] // batch_size
  for _, batch in zip(tqdm.trange(steps), ds_test.as_numpy_iterator()):
    predicted = vit_apply_repl(params_repl, batch['image'])
    is_same = predicted.argmax(axis=-1) == batch['label'].argmax(axis=-1)
    good += is_same.sum()
    total += len(is_same.flatten())
  return good / total
[ ]
# Random performance without fine-tuning.
get_accuracy(params_repl)
INFO:absl:Load dataset info from /root/tensorflow_datasets/cifar10/3.0.2
100%|██████████| 19/19 [01:07<00:00,  3.58s/it]
DeviceArray(0.10063734, dtype=float32)
Fine-tune
[ ]
# 100 Steps take approximately 15 minutes in the TPU runtime.
total_steps = 100
warmup_steps = 5
decay_type = 'cosine'
grad_norm_clip = 1
# This controls in how many forward passes the batch is split. 8 works well with
# a TPU runtime that has 8 devices. 64 should work on a GPU. You can of course
# also adjust the batch_size above, but that would require you to adjust the
# learning rate accordingly.
accum_steps = 8
base_lr = 0.03
[ ]
# Check out train.make_update_fn in the editor on the right side for details.
lr_fn = utils.create_learning_rate_schedule(total_steps, base_lr, decay_type, warmup_steps)
update_fn_repl = train.make_update_fn(
    apply_fn=model.apply, accum_steps=accum_steps, lr_fn=lr_fn)
# We use a momentum optimizer that uses half precision for state to save
# memory. It als implements the gradient clipping.
opt = momentum_clip.Optimizer(grad_norm_clip=grad_norm_clip).create(params)
opt_repl = flax.jax_utils.replicate(opt)
/usr/local/lib/python3.7/dist-packages/jax/lib/xla_bridge.py:317: UserWarning: jax.host_count has been renamed to jax.process_count. This alias will eventually be removed; please update your code.
  "jax.host_count has been renamed to jax.process_count. This alias "
/usr/local/lib/python3.7/dist-packages/jax/lib/xla_bridge.py:304: UserWarning: jax.host_id has been renamed to jax.process_index. This alias will eventually be removed; please update your code.
  "jax.host_id has been renamed to jax.process_index. This alias "
[ ]
# Initialize PRNGs for dropout.
update_rng_repl = flax.jax_utils.replicate(jax.random.PRNGKey(0))
/usr/local/lib/python3.7/dist-packages/jax/lib/xla_bridge.py:317: UserWarning: jax.host_count has been renamed to jax.process_count. This alias will eventually be removed; please update your code.
  "jax.host_count has been renamed to jax.process_count. This alias "
/usr/local/lib/python3.7/dist-packages/jax/lib/xla_bridge.py:304: UserWarning: jax.host_id has been renamed to jax.process_index. This alias will eventually be removed; please update your code.
  "jax.host_id has been renamed to jax.process_index. This alias "
[ ]
losses = []
lrs = []
# Completes in ~20 min on the TPU runtime.
for step, batch in zip(
    tqdm.trange(1, total_steps + 1),
    ds_train.as_numpy_iterator(),
):

  opt_repl, loss_repl, update_rng_repl = update_fn_repl(
      opt_repl, flax.jax_utils.replicate(step), batch, update_rng_repl)
  losses.append(loss_repl[0])
  lrs.append(lr_fn(step))

plt.plot(losses)
plt.figure()
plt.plot(lrs)

[ ]
# Should be ~96.7% for Mixer-B/16 or 97.7% for ViT-B/32 on CIFAR10 (both @224)
get_accuracy(opt_repl.target)
INFO:absl:Load dataset info from /root/tensorflow_datasets/cifar10/3.0.2
100%|██████████| 19/19 [00:32<00:00,  1.73s/it]
DeviceArray(0.9762541, dtype=float32)
Inference
[ ]
# Download a pre-trained model.

if model_name.startswith('Mixer'):
  # Download model trained on imagenet2012
  ![ -e "$model_name"_imagenet2012.npz ] || gsutil cp gs://mixer_models/imagenet1k/"$model_name".npz "$model_name"_imagenet2012.npz
  model = models.MlpMixer(num_classes=1000, **model_config)
else:
  # Download model pre-trained on imagenet21k and fine-tuned on imagenet2012.
  ![ -e "$model_name"_imagenet2012.npz ] || gsutil cp gs://vit_models/imagenet21k+imagenet2012/"$model_name".npz "$model_name"_imagenet2012.npz
  model = models.VisionTransformer(num_classes=1000, **model_config)

import os
assert os.path.exists(f'{model_name}_imagenet2012.npz')
[ ]
# Load and convert pretrained checkpoint.
params = checkpoint.load(f'{model_name}_imagenet2012.npz')
params['pre_logits'] = {}  # Need to restore empty leaf for Flax.
[ ]
# Get imagenet labels.
!wget https://storage.googleapis.com/bit_models/ilsvrc2012_wordnet_lemmas.txt
imagenet_labels = dict(enumerate(open('ilsvrc2012_wordnet_lemmas.txt')))
--2021-06-20 16:44:59--  https://storage.googleapis.com/bit_models/ilsvrc2012_wordnet_lemmas.txt
Resolving storage.googleapis.com (storage.googleapis.com)... 74.125.142.128, 74.125.20.128, 74.125.197.128, ...
Connecting to storage.googleapis.com (storage.googleapis.com)|74.125.142.128|:443... connected.
HTTP request sent, awaiting response... 200 OK
Length: 21675 (21K) [text/plain]
Saving to: ‘ilsvrc2012_wordnet_lemmas.txt.1’

ilsvrc2012_wordnet_ 100%[===================>]  21.17K  --.-KB/s    in 0s      

2021-06-20 16:44:59 (135 MB/s) - ‘ilsvrc2012_wordnet_lemmas.txt.1’ saved [21675/21675]

[ ]
# Get a random picture with the correct dimensions.
resolution = 224 if model_name.startswith('Mixer') else 384
!wget https://picsum.photos/$resolution -O picsum.jpg
import PIL
img = PIL.Image.open('picsum.jpg')
img

[ ]
# Predict on a batch with a single item (note very efficient TPU usage...)
logits, = model.apply(dict(params=params), (np.array(img) / 128 - 1)[None, ...], train=False)
[ ]
preds = flax.nn.softmax(logits)
for idx in preds.argsort()[:-11:-1]:
  print(f'{preds[idx]:.5f} : {imagenet_labels[idx]}', end='')
0.13330 : sandbar, sand_bar
0.09332 : seashore, coast, seacoast, sea-coast
0.05257 : jeep, landrover
0.05188 : Arabian_camel, dromedary, Camelus_dromedarius
0.01251 : horned_viper, cerastes, sand_viper, horned_asp, Cerastes_cornutus
0.00753 : tiger_beetle
0.00744 : dung_beetle
0.00711 : sidewinder, horned_rattlesnake, Crotalus_cerastes
0.00703 : leatherback_turtle, leatherback, leathery_turtle, Dermochelys_coriacea
0.00647 : pole
11511311411011111210810910710510610410310210110099989796959493
  image_height = image['height']
  image_width = image['width']
  filename = image['file_name']
  image_id = image['id']

  full_path = os.path.join(image_dir, filename)
  with tf.io.gfile.GFile(full_path, 'rb') as fid:
    encoded_jpg = fid.read()
  encoded_jpg_io = io.BytesIO(encoded_jpg)
  image = PIL.Image.open(encoded_jpg_io)


check
22 秒
完成时间：15:29
