import torch
import torchvision

frames = torch.load("./frames.pt")
torchvision.io.write_video("frames.mp4", frames, 10)

frames = torch.load("./frames_post.pt")
torchvision.io.write_video("frames_post.mp4", frames, 10)


frames = torch.load("./frames_post_255.pt")
torchvision.io.write_video("frames_post_255.mp4", frames, 10)