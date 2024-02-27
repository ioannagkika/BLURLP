import predict_video
import predict_image
import yaml
import glob
import pathlib

with open('options.yaml', 'r') as file:
    opts = yaml.safe_load(file)
    
image_ext = [".bmp", ".dng", ".jpeg", ".jpg", ".mpo", ".png", "tif", "tiff", ".webp", ".pfm"]
video_ext = [".asf", ".avi", ".gif", ".m4v", ".mkv", ".mov", ".mp4", ".mpeg", ".mpg", ".ts", ".wmv", ".webm"]

images = [f for f_ in [glob.glob(opts['input_path']+"*"+e) for e in image_ext] for f in f_]
for image in images:
    predict_image.blur_in_image(image_in = image, image_out = opts['output_path'] + pathlib.Path(image).stem +".png",
                                show = opts['show'], save_image=opts['save'])
       
videos = [f for f_ in [glob.glob(opts['input_path']+"*"+e) for e in video_ext] for f in f_]
for video in videos:
    predict_video.blur_in_video(video_in = video, video_out = opts['output_path'] + pathlib.Path(video).stem +".avi", 
                                show = opts['show'], save_video = opts['save'])