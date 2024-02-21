import predict_video
import predict_image
import yaml
import glob
import pathlib

with open('paths.yaml', 'r') as file:
    paths = yaml.safe_load(file)
    
image_ext = [".bmp", ".dng", ".jpeg", ".jpg", ".mpo", ".png", "tif", "tiff", ".webp", ".pfm"]
video_ext = [".asf", ".avi", ".gif", ".m4v", ".mkv", ".mov", ".mp4", ".mpeg", ".mpg", ".ts", ".wmv", ".webm"]

images = [f for f_ in [glob.glob(paths['input_image_path']+"*"+e) for e in image_ext] for f in f_]
for image in images:
    predict_image.blur_in_image(image_in = image, image_out = paths['output_image_path'] + pathlib.Path(image).stem +".png",show = False, save_image=True)
       
videos = [f for f_ in [glob.glob(paths['input_video_path']+"*"+e) for e in video_ext] for f in f_]
for video in videos:
    predict_video.blur_in_video(video_in = video, video_out = paths['output_video_path'] + pathlib.Path(video).stem +".avi", show = False, save_video = True)