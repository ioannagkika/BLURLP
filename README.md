# Licence plate blurring using YOLOv8_obb 

This is my WiP project for licence plate detecting and blurring in images and videos using yolov8 oriented bounding boxes. This project is just a hobby project and is not intended for comercial use.

## Environment

The code has been developed on:
* Ubuntu: 22.04
* CUDA Version: 11.5
* Python 3.10.12

## Dependencies

You can install the requirements with the command: 
```bash
pip install -r requirements.txt
```

## Pretrained Weights & Dataset

You can download the weights of the pre-trained model from [Google Drive](https://drive.google.com/drive/folders/1HHt-oDnS-wRuDBi4tJiqLaAwtfUqM-F8?usp=sharing) and place into the folder weights.

The dataset used for training is a subset of the [CCPD dataset](https://github.com/detectRecog/CCPD).

## Inference

For inference, run following command. 

```bash
python main.py
```
Make sure you have added the proper paths in the options.yaml.
* weights: the path of the weights.pt file
* input_path: the path of the folder where the images and/or videos you want to test are
* output_path: the path where you want to save the output of the model
* save: if you want to save the output of the model keep it to True, else change it to False.
* show: if you want to preview the results change it to True, else keep it to False.

## Example

[![Watch the video](https://github.com/ioannagkika/licence_plate_blurring_yolov8_obb/blob/main/output/example.mp4)

Note: The output video will be in .avi format. If you want to convert it to .mp4 run the following command on terminal:

```bash
ffmpeg -i example.avi example.mp4
```

