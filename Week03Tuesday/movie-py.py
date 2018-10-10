
from moviepy.editor import *

img = []
for el in range(100):
    img.append(str(el) + '.png')
    
clips = ImageSequenceClip(img, fps=24)
clips.write_videofile("test.mp4", fps=24)