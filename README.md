Code for "One Token for Detecting Various Changes" (Under Review) is being updated.

## 环境
Ubuntu 20.04

python==3.7

pytorch==1.9.0

timm==0.4.5


## 数据集和模型
处理后的VL-CMU-CD二分类变化检测数据集：[CMUBi](https://pan.baidu.com/s/1UI34-Wide_pVFBCX9eaQpQ?pwd=8sgw)

模型权重：

| 模型     | F1     |
| ---------- | ---------- |
| [CSST_Siam_T_DSIFN](https://pan.baidu.com/s/1_CdnHdOUYCENeHf2MIdbEg?pwd=4ba7) | 94.81 |
| [CSST_Siam_T_CDD](https://pan.baidu.com/s/1Coj_MwMmT8HZljtwGikB5g?pwd=7r3n) | 97.49 |
| [CSST_ChangeFormer_CDD](https://pan.baidu.com/s/1sdtbXrmATqcHbWyVB685BA?pwd=rmf1) | 97.75 |
| [CSST_Siam_T_LEVIR](https://pan.baidu.com/s/1ROvAKzMvqgBICIVN99Ixsw?pwd=gxup) | 91.53 |
| [CSST_VGG_LEVIR](https://pan.baidu.com/s/1xIMtXjNh4AAbX3e06WUFJA?pwd=v5g9) | 91.57 |
| [CSST_UNet++_LEVIR](https://pan.baidu.com/s/1hNYsSQl00kWM6pOyqzOSYw?pwd=845w) | 91.63 |
| [CSST_Siam_T_CMUBi](https://pan.baidu.com/s/1Td8i5YoNtswYplfpf5OnZg?pwd=e3hc) | 64.69 |
| [CSST_Siam_RPT_CMUBi](https://pan.baidu.com/s/1tgrl3ixt5e-qCPOX-Tkhrg?pwd=cmrx) | 77.77 |


## 运行

- 训练模型

对 run_cd.sh文件中各项参数进行相应更改后在终端运行

```sh
bash run_cd.sh
```

- 测试
  
对 eval_cd.py文件中的--project_name，--net_G，--data_name等参数进行相应更改后在终端运行

```sh
python eval_cd.py
```
