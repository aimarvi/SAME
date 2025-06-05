import os
import pickle

from PIL import Image

import torch
from torchvision import models, transforms
from tqdm import tqdm

# Load pretrained AlexNet
alexnet = models.alexnet(pretrained=True)
alexnet.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),           # Resize to AlexNet input
    transforms.ToTensor(),                   # Convert to tensor
    transforms.Normalize(                    # Normalize like ImageNet
        mean=[0.485, 0.456, 0.406],          
        std=[0.229, 0.224, 0.225]
    )
])

# Store activations in a dictionary
features = {}
handles = []

# Define hook
def get_activation(name):
    def hook(model, input, output):
        features[name] = output.detach().cpu()
    return hook

target_layers = {
    'conv1': 0,
    'conv2': 3,
    'fc6': 4,
    'fc7': 5,
}

for layer,ind in target_layers.items():
    handle = alexnet.features[ind].register_forward_hook(get_activation(layer))
    handles.append(handle)


for cat in tqdm(['same_label', 'same_image', 'different_label']):
    stim_dir = os.path.join('./mnist_stim', cat) 
    all_features = {cat: []}

    for fname in sorted(os.listdir(stim_dir)):
        if not fname.endswith('.png'):
            continue
        # MNIST is grayscale, so convert to RGB
        img = Image.open(os.path.join(stim_dir, fname)).convert("RGB")
        img_tensor = transform(img).unsqueeze(0)  # add batch dim

        with torch.no_grad():
            _ = alexnet(img_tensor)

        # Save features for this image
        feature_vec = {
            'stimulus': fname,
            'conv1': features['conv1'].flatten().numpy(),
            'conv2': features['conv2'].flatten().numpy(),
            'fc6': features['fc6'].flatten().numpy(),
            'fc7': features['fc7'].flatten().numpy()
        }

        all_features[cat].append(feature_vec)

    for h in handles:
        h.remove()

with open('mnist_features.pkl', 'wb') as f:
    pickle.dump(all_features, f)
