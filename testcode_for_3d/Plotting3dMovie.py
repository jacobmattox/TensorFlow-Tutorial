from moviepy.editor import *

img = []
for el in range(0, 360, 10):
    img.append('rotate_' + str(el) + '.png')

clips = ImageSequenceClip(img, fps=5)
clips.write_videofile("test.mp4", fps=5)