
<!-- ```bash
git clone 
cd BEDM
``` -->


# 1. CIFAR-100

## Fetch Data
```bash
mkdir -p exp/CIFAR100
wget -P exp/CIFAR100 https://www.cs.utoronto.ca/~kriz/cifar-100-python.tar.gz
tar -zxvf exp/CIFAR100/cifar-100-python.tar.gz -C exp/CIFAR100
```

## Run
```bash
python run_cifar.py
```

# 2. ImageNet

## Fetch Data
Modify the `root` field in `options/imgnet100.yaml` and `options/imgnet1000.yaml`.

The `root` directory should in this order.
```
├── root/
│ ├── train/
│ │      ├── n01440764/
│ │      ├── n01443537/
│ │      └── .../
│ └── val/
│        ├── n01440764/
│        ├── n01443537/
│        └── .../
```


## Run
```bash
# ImageNet-100
python run_imgnet.py  --dataset  imgnet100

# ImageNet-1000
python run_imgnet.py  --dataset  imgnet1000
```



