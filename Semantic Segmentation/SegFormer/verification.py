import mmcv

from mmseg.apis import inference_segmentor, init_segmentor

config_file = 'local_configs/segformer/B4/segformer.b4.1024x1024.city.160k.py'
checkpoint_file = 'checkpoints/segformer.b4.1024x1024.city.160k.pth'

# build the model from a config file and a checkpoint file
model = init_segmentor(config_file, checkpoint_file, device='cuda:0')

# test a single image and show the results
# img = 'out.jpg'  # or img = mmcv.imread(img), which will only load it once
# result = inference_segmentor(model, img)
# visualize the results in a new window
# model.show_result(img, result, show=True)
# or save the visualization results to image files
# model.show_result(img, result, out_file='result.jpg')

video = mmcv.VideoReader('Daytime.mp4')  # input
for idx, frame in enumerate(video):
    result = inference_segmentor(model, frame)
    model.show_result(frame, result, out_file='frames_day/{:06d}.jpg'.format(idx))

mmcv.frames2video('frames_day', 'Day_Seg.avi')  # output
