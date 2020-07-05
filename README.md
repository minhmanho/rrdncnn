# Down-Sampling Based Video Coding with Degradation-aware Restoration-Reconstruction Deep Neural Network
[Man M. Ho](https://minhmanho.github.io/), Gang He, Zheng Wang, and [Jinjia Zhou](https://www.zhou-lab.info/jinjia-zhou).

## News

| Date        | News                                                      |
| ----------- | --------------------------------------------------------- |
| 2020/07/05  | We have updated the evaluation code for RR-DnCNN and RR-DnCNN v2.0|
| 2020/01/12  | Our RR-DnCNN will be published together with its version 2.0 at once|
| 2020/01/07  | We won **Best Paper Runner-up Award** at MMM2020 |
| 2019/09/24  | Our work has been accepted to MMM2020 (Oral) |

## Tasks
- [ ] Prepare HTML webpage version.
- [ ] Summarize this work here.
- [x] Publish evaluation code.

## Overview:
[![](http://img.youtube.com/vi/-oNjWXAM5Hc/0.jpg)](http://www.youtube.com/watch?v=-oNjWXAM5Hc "Click for watching demonstration")

## Prerequisites

- Ubuntu 16.04
- OpenCV
- [PyTorch](https://pytorch.org/) >= 1.1.0
- Numpy

## Getting Started.

Let's reproduce our result.

### 1. Clone this repo:
```
git clone https://github.com/minhmanho/rrdncnn.git
cd rrdncnn
```

### 2. Prepare data:

Since the files are large, URLs here needs your confirmation; otherwise, just run the command lines with recursion '-r'.

\+ Johnny 1280x720 in Class E (65 MB) (_recommended_).
```
wget --no-check-certificate -r 'https://docs.google.com/uc?export=download&id=1PZWTdTkMxLpaSTJ4A7Xxw2QK6tmYy-d9' -O ./data/class_e.zip
unzip -o ./data/class_e.zip -d ./data/
```
*or*
```
./data/download_class_e.sh
```

\+ BasketballDrill 832x480 in Class C (384 MB) (_optional_).
```
wget --no-check-certificate -r 'https://docs.google.com/uc?export=download&id=1GsnzTPGEVS8v-aMfQyrKExqKx-jZP7uP' -O ./data/class_c.zip
unzip -o ./data/class_c.zip -d ./data/
```
*or*
```
./data/download_class_c.sh
```

The expected data structure (for Johnny 1280x720 compressed by HEVC 16.20 with Low Delay P - LDP) is as:

```
./data/class_e
│   DHR_LDP.csv (HEVC on HR as <video_name>,<bir-rate kbps>, <Y-PSNR>, <U-PSNR>, <V-PSNR>, <YUV-PSNR>)
│   DLR_LDP.csv (HEVC on LR as <video_name>,<bir-rate kbps>, <Y-PSNR>, <U-PSNR>, <V-PSNR>, <YUV-PSNR>)
│   DHR_LDP_QP[32-47].txt (HEVC running logs)
|   DLR_LDP_QP[32-47].txt (HEVC running logs)
|
└───DLR_LDP_QP[32-47] (Decoded Low-Resolution)
│   │   Johnny_1280x720_50.yuv
|
└───HR (Uncompressed High-Resolution)
│   │   Johnny_1280x720_50.yuv
|
└───LR (Uncompressed Low-Resolution down-sampled x2 by bicubic)
│   │   Johnny_1280x720_50.yuv
```

### 3. Run models:
```
CUDA_VISIBLE_DEVICES=0 python eval.py --vin <path/to/class/folder> --ckpt <path/to/model> --size <video_size HxW> --decoder_folder <folder/contains/videos> --log_dir ./logs/
```

For example:

\+ Run with RR-DnCNN.
```
CUDA_VISIBLE_DEVICES=0 python eval.py --vin ./data/class_e/ --ckpt ./models/rrdncnn.pth.tar --size 360x640 --decoder_folder DLR_LDP_QP32
```
\+ Run with RR-DnCNN v2.0.
```
CUDA_VISIBLE_DEVICES=0 python eval.py --vin ./data/class_e/ --ckpt ./models/rrdncnn_v2.pth.tar --size 360x640 --decoder_folder DLR_LDP_QP32
```

\+ Run with our prepared script.
```
./run.sh
```

## Results

After running, you will get the results as follows:
+ \<path/to/class/folder\> or <args.out> /\<model\>\_SDLR\_\<configuration\>\_\<QP\> (Folder contains reconstructed videos)
+ \<args.log_dir\>/\<configuration\>\_\<model\>.csv (CSV file contains quantitative results)

Beside printing, we also export the results in CSV to *log_dir* for copying/pasting, where the columns (*left-to-right*) in the CSV file represent:
```
video_name_QP, HR-Y-PSNR, HR-Y-SSIM, LR-Y-PSNR, LR-Y-SSIM
```
E.g., 
```
rrdncnn_Johnny_1280x720_50_DLR_LDP_QP32,35.0897,0.9118,38.2423,0.9594
```

Finally, we calculate the BD-rates based on the results of HEVC (attached in the class package) and our methods. The expected output should be:

<table class="tg">
<thead>
  <tr>
    <th class="tg-nrix" rowspan="2">Sequence_QP</th>
    <th class="tg-nrix" colspan="2">HEVC&nbsp;16.20</th>
    <th class="tg-nrix" rowspan="2">Low Bit-rate (kbps)</th>
    <th class="tg-nrix" colspan="2">RR-DnCNN</th>
    <th class="tg-nrix" colspan="2">RR-DnCNN&nbsp;v2.0</th>
  </tr>
  <tr>
    <td class="tg-nrix">Bit-rate (kbps)</td>
    <td class="tg-nrix">PSNR (dB)</td>
    <td class="tg-nrix">PSNR</td>
    <td class="tg-nrix">BD-BR (%)</td>
    <td class="tg-nrix">PSNR (dB)</td>
    <td class="tg-nrix">BD-BR (%)</td>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-7zrl">Johnny_1280x720_50.yuv_32</td>
    <td class="tg-7zrl">87.7059</td>
    <td class="tg-7zrl">38.9999</td>
    <td class="tg-7zrl">34.4353</td>
    <td class="tg-7zrl">35.0897</td>
    <td class="tg-nrix" rowspan="4">-12.9307</td>
    <td class="tg-7zrl">35.4468</td>
    <td class="tg-nrix" rowspan="4">-15.8121</td>
  </tr>
  <tr>
    <td class="tg-7zrl">Johnny_1280x720_50.yuv_37</td>
    <td class="tg-7zrl">46.9529</td>
    <td class="tg-7zrl">36.7226</td>
    <td class="tg-7zrl">18.9812</td>
    <td class="tg-7zrl">33.4981</td>
    <td class="tg-7zrl">33.6468</td>
  </tr>
  <tr>
    <td class="tg-7zrl">Johnny_1280x720_50.yuv_42</td>
    <td class="tg-7zrl">25.5059</td>
    <td class="tg-7zrl">34.1018</td>
    <td class="tg-7zrl">11.4706</td>
    <td class="tg-7zrl">31.2232</td>
    <td class="tg-7zrl">31.3090</td>
  </tr>
  <tr>
    <td class="tg-7zrl">Johnny_1280x720_50.yuv_47</td>
    <td class="tg-7zrl">14.9129</td>
    <td class="tg-7zrl">31.5612</td>
    <td class="tg-7zrl">7.0847</td>
    <td class="tg-7zrl">28.7739</td>
    <td class="tg-7zrl">28.8354</td>
  </tr>
  <tr>
    <td class="tg-7zrl">BasketballDrill_832x480_50.yuv_32</td>
    <td class="tg-7zrl">1833.6642</td>
    <td class="tg-7zrl">35.4191</td>
    <td class="tg-7zrl">685.0295</td>
    <td class="tg-7zrl">31.0428</td>
    <td class="tg-nrix" rowspan="4">-10.9197</td>
    <td class="tg-7zrl">31.0627</td>
    <td class="tg-nrix" rowspan="4">-12.4616</td>
  </tr>
  <tr>
    <td class="tg-7zrl">BasketballDrill_832x480_50.yuv_37</td>
    <td class="tg-7zrl">1018.0177</td>
    <td class="tg-7zrl">32.7758</td>
    <td class="tg-7zrl">392.1459</td>
    <td class="tg-7zrl">29.2932</td>
    <td class="tg-7zrl">29.365</td>
  </tr>
  <tr>
    <td class="tg-7zrl">BasketballDrill_832x480_50.yuv_42</td>
    <td class="tg-7zrl">559.657</td>
    <td class="tg-7zrl">30.1624</td>
    <td class="tg-7zrl">208.8544</td>
    <td class="tg-7zrl">26.9593</td>
    <td class="tg-7zrl">27.0212</td>
  </tr>
  <tr>
    <td class="tg-7zrl">BasketballDrill_832x480_50.yuv_47</td>
    <td class="tg-7zrl">277.0484</td>
    <td class="tg-7zrl">27.5161</td>
    <td class="tg-7zrl">97.052</td>
    <td class="tg-7zrl">24.7342</td>
    <td class="tg-7zrl">24.7411</td>
  </tr>
</tbody>
</table>


Let's check our pre-run version (26 KB) for more details.
```
wget 'https://docs.google.com/uc?export=download&id=1VkiMalZp_UxejxA3ScoE7fzDDZoWACFc' -O ./logs/result_sample.zip
unzip -o ./logs/result_sample.zip -d ./logs/
```
*or*
```
./logs/download_result_sample.sh
```

## Citations
Please cite our works if you find them helpful.
```
@inproceedings{ho2020down,
  title={Down-Sampling Based Video Coding with Degradation-Aware Restoration-Reconstruction Deep Neural Network},
  author={Ho, Minh-Man and He, Gang and Wang, Zheng and Zhou, Jinjia},
  booktitle={International Conference on Multimedia Modeling},
  pages={99--110},
  year={2020},
  organization={Springer}
}
```
```
@article{ho2020rr,
  title={RR-DnCNN v2.0: Enhanced Restoration-Reconstruction Deep Neural Network for Down-Sampling Based Video Coding},
  author={Ho, Man M and Zhou, Jinjia and He, Gang},
  journal={arXiv preprint arXiv:2002.10739},
  year={2020}
}
```

## License

This repository (as well as its materials) is for non-commercial uses and research purposes only.


## Acknowledgement

Thank G. Bjontegaard and S. Pateux for [ETRO's Bjontegaard Metric Implementation](https://github.com/tbr/bjontegaard_etro).
```
G. Bjontegaard, Calculation of average PSNR differences between RD-curves (VCEG-M33)
S. Pateux, J. Jung, An excel add-in for computing Bjontegaard metric and its evolution
```
This work is supported by JST, PRESTO Grant Number JPMJPR1757 Japan.

## Contact
If you have any suggestions, questions, or the use of materials infringes your copyright/license, please contact me <man.hominh.6m@stu.hosei.ac.jp>.
