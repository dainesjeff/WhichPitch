import cv2
import json
import numpy as np
from torchvision import datasets, transforms
import torch
from torch.utils.data import Dataset

f = open('config.json')
config = json.load(f)

class CustomImageDataset(Dataset):
    def __init__(self, labels, frames, transform, target_transform=None):
        self.img_labels = labels
        self.frames = frames
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        image = frames[idx]
        label = self.img_labels[idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label



cap = cv2.VideoCapture(config["TestFilePath"])



frames = []




while(1):
   ret, frame = cap.read(0)
   if frame is not None:
       #gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
       #resized_gray_image = cv2.resize(gray_image, (256, 256))
       resized_image = cv2.resize(frame, (256, 256))
       #cv2.resize(gray_image, (0, 0), fx = config["FrameScaleFactor"], fy =config["FrameScaleFactor"])
       frames.append(resized_image)
   if cv2.waitKey(1) & 0xFF == ord('q') or ret==False :
       cap.release()
       break

labels = [0] * len(frames)

tc = transforms.Compose([
        transforms.ToTensor()
    ])

dset = CustomImageDataset(labels, frames, tc)

dloader = torch.utils.data.DataLoader(dset, batch_size=10, shuffle=False)

# fetch pretrained model
model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)

pooling_layer = model._modules.get('avgpool')

def copy_embeddings(m, i, o):
    """Copy embeddings from the penultimate layer.
    """
    o = o[:, :, 0, 0].detach().numpy().tolist()
    outputs.append(o)

outputs = []
# attach hook to the penulimate layer
_ = pooling_layer.register_forward_hook(copy_embeddings)

print(dloader)

for X, y in dloader:
    _ = model(X)

list_embeddings = [item for sublist in outputs for item in sublist]

print(len(list_embeddings)) # returns 918
print(np.array(list_embeddings[0]).shape)

#frames = np.asarray(frames)
#print(frames.shape)
