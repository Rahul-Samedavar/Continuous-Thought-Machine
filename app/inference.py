import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms

CLASS_LIST = ["glioma", "meningioma", "notumor", "pituitary"]

DATASET_MEAN = [0.23367126286029816, 0.23367522656917572, 0.23372678458690643]
DATASET_STD = [0.1767614781856537, 0.1767629235982895, 0.17677341401576996]

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=DATASET_MEAN, std=DATASET_STD)
])

def run_inference(image_path, model, output_dir="outputs"):
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(next(model.parameters()).device)

    with torch.no_grad():
        predictions, certainties, _, _, _, attention_tracking = model(input_tensor, track=True)

    probs = torch.softmax(predictions[0, :, -1], dim=-1).cpu().numpy()
    predicted_idx = int(np.argmax(probs))

    os.makedirs(output_dir, exist_ok=True)
    gif_path = os.path.join(output_dir, os.path.basename(image_path) + "_viz.gif")

    from app.visualization import make_gif
    make_gif(
        input_tensor,
        predictions,
        certainties,
        attention_tracking,
        predicted_idx,
        DATASET_MEAN,
        DATASET_STD,
        CLASS_LIST,
        gif_path
    )

    return {
        "predicted_class": CLASS_LIST[predicted_idx],
        "confidence": float(probs[predicted_idx]),
        "class_probabilities": dict(zip(CLASS_LIST, probs.tolist())),
        "certainty_curve": certainties[0, 1].cpu().tolist(),
        "gif_filename": os.path.basename(gif_path)
    }
