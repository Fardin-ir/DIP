from moviepy.editor import *

clip_a = VideoFileClip("a.mp4")
clip_b = VideoFileClip("b.mp4")
clip_c = VideoFileClip("c.mp4")
clip_d = VideoFileClip("d.mp4")
clip_e = VideoFileClip("e.mp4")

final = concatenate_videoclips([clip_a, clip_b, clip_c, clip_d, clip_e])

final.ipython_display()


