Code for "One Token for Detecting Various Changes" (Under Review) is being updated.

## Requirement
Ubuntu 20.04

python==3.7

pytorch==1.9.0

timm==0.4.5


## Dataset and Weights
The VL-CMU-CD (binary classification) data set：[CMUBi](https://pan.baidu.com/s/1UI34-Wide_pVFBCX9eaQpQ?pwd=8sgw)

CSST-based Models：

| Model Weights     | F1     |
| ---------- | ---------- |
| [CSST_Siam_T_DSIFN](https://pan.baidu.com/s/1_CdnHdOUYCENeHf2MIdbEg?pwd=4ba7) | 94.81 |
| [CSST_Siam_T_CDD](https://pan.baidu.com/s/1Coj_MwMmT8HZljtwGikB5g?pwd=7r3n) | 97.49 |
| [CSST_ChangeFormer_CDD](https://pan.baidu.com/s/1sdtbXrmATqcHbWyVB685BA?pwd=rmf1) | 97.75 |
| [CSST_Siam_T_LEVIR](https://pan.baidu.com/s/1ROvAKzMvqgBICIVN99Ixsw?pwd=gxup) | 91.53 |
| [CSST_VGG_LEVIR](https://pan.baidu.com/s/1xIMtXjNh4AAbX3e06WUFJA?pwd=v5g9) | 91.57 |
| [CSST_UNet++_LEVIR](https://pan.baidu.com/s/1hNYsSQl00kWM6pOyqzOSYw?pwd=845w) | 91.63 |
| [CSST_Siam_T_CMUBi](https://pan.baidu.com/s/1Td8i5YoNtswYplfpf5OnZg?pwd=e3hc) | 64.69 |
| [CSST_Siam_RPT_CMUBi](https://pan.baidu.com/s/1tgrl3ixt5e-qCPOX-Tkhrg?pwd=cmrx) | 77.77 |


## Usage

- Training

	- Set the necessary configures in file 'run_cd.sh', then run it in the terminal.

```sh
bash run_cd.sh
```

	- Training CSST_ChangeFormer

We used [ChangeFormer's project] (https://github.com/wgcban/ChangeFormer) to train CSST_ChangeFormer. 

Put the file 'csstchangeformer.py' into the subfolder 'models' of the project of ChangeFormer. Set corresponding configs in run_cd.sh, training epochs=200. optimizer=adamw, weight decay=0.01, lr=2e−4, batchsize=16, splitval=val.

Run the script in the terminal.

- Testing
  
Set the necessary configs, such as --project_name，--net_G，--data_name, and etc, in file 'eval_cd.py'. Then, run it in the terminal.

```sh
python eval_cd.py
```

##Acknowledgment

Appreciate for the two code works:

(https://github.com/justchenhao/BIT_CD)

(https://github.com/wgcban/ChangeFormer)

