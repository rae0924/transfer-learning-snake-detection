from model import SnakeDetector
import torch
from torch import nn
from torchvision import transforms
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

# load trained model
detector = SnakeDetector()
proj_dir = os.path.dirname(__file__)
model_path = os.path.join(proj_dir, 'save/snake_detector.pt')
state_dict = torch.load(model_path)
detector.load_state_dict(state_dict)

# preprocess
image_path = os.path.join(proj_dir, 'save/example_9.jpg')
image = (cv2.imread(image_path)[:,:,::-1])
image = cv2.resize(image, (224,224))
x = torch.tensor(image, dtype=torch.float)
x = x/255
x = x.permute(2,0,1)
normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
x = normalize(x)
x = x.unsqueeze(0)

# enable gradients
for param in detector.features.parameters():
    param.requires_grad = True

# gradient class activation maps generation
detector.eval()
fmap = detector.features[:29]
logit = detector(x)
activations = fmap(x)
detector.zero_grad()
logit.backward(retain_graph=True)
pooled_grads = detector.features[28].weight.grad.data.mean((1,2,3))
for i in range(activations.shape[1]):
    activations[:,i,:,:] *= pooled_grads[i]
map = torch.mean(activations, dim=1)[0].detach()

# heat map generation
SZ = 224
m,M = map.min(), map.max()
map = 255 * ((map-m) / (M-m))  # normalize
map = np.uint8(map)
map = cv2.resize(map, (SZ,SZ))
map = cv2.applyColorMap(255-map, cv2.COLORMAP_JET)
map = np.uint8(map)
map = np.uint8(map*0.5 + image*0.5)

# predicted results
out = logit.round()
if out == 1:
    prediction = 'snake'
else:
    prediction = 'not snake'

# plot image vs. CAM
f, (ax1, ax2) = plt.subplots(1, 2)
f.suptitle(f'prediction: {prediction}', y=0.82)
ax1.imshow(image)
ax2.imshow(map)
f.savefig(os.path.join(proj_dir, 'save/cam_9.png'))