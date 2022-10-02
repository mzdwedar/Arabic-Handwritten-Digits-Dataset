# Arabic-Handwritten-Digits-Dataset

![pic01](https://user-images.githubusercontent.com/28225321/154095807-669138c9-527e-4a07-9c32-2d49ed7cb941.jpg)

## Install requirements

```
! pip3 install -r requirements.txt
```

## Dataset

[Arabic Handwritten Digits Dataset](https://www.kaggle.com/mloey1/ahdd1)

### **Download the dataset using Kaggle api**

after downloading the credentials

```
! pip install -q kaggle

! mkdir ~/.kaggle

! cp kaggle.json ~/.kaggle/

! chmod 600 ~/.kaggle/kaggle.json

! kaggle  datasets download -d datasets mloey1/ahdd1
```

### **Extract the dataset**

```
python3 utils/extract_data.py
```

## Train

```
python3 train.py
```

## From pytorch to ONNX

```
python3 to_onnx.py
```

available arguments:

- `--path`: pytorch model Path. Default= 'saved_model/AHDD_resnet18.pt'

## Inference with ONNX

```
python3 onnx_infer.py --image_path path
```

available arguments:

- `--model_path`: pytorch model Path. Default= 'saved_model/AHDD_resnet18.onnx'
- `--image_path`: image Path.

## Model

  `ResNet18`

## Framework

  `pytorch`
