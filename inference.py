import os
import torch

from models.ctm import ContinuousThoughtMachine



import torch
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from models.ctm import ContinuousThoughtMachine as CTM
import urllib
from IPython.display import Image as IPyImage, display
from pprint import pprint
import numpy as np
import seaborn as sns
import os
import torch.nn.functional as F
from matplotlib import patheffects
from scipy import ndimage
import imageio
import mediapy
from tasks.image_classification.plotting import plot_neural_dynamics

log_dir = 'logs/custom_run'
data_root = 'dataset/'
batch_size = 32
batch_size_test = 32
lr = 1e-3
training_iterations = 10001
warmup_steps = 500
use_scheduler = True
scheduler_type = 'cosine'
milestones = [8000, 15000, 20000]
gamma = 0.1
weight_decay = 0.0
gradient_clipping = -1
num_workers_train = 1
num_workers_test = 1
save_every = 1000
seed = 412
track_every = 1000
n_test_batches = 20
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
use_amp = False

d_model = 512
dropout = 0.0
backbone_type = 'resnet18-4'
d_input = 128
heads = 4
iterations = 75
positional_embedding_type = 'none'
synapse_depth = 4
n_synch_out = 512
n_synch_action = 512
neuron_select_type = 'random-pairing'
n_random_pairing_self = 0
memory_length = 25
deep_memory = True
memory_hidden_dims = 4
dropout_nlm = None
do_normalisation = False

prediction_reshaper = [-1]
out_dims = 4

checkpoint = torch.load(r'logs\custom_run\checkpoint_10000.pt', map_location=device)

model = ContinuousThoughtMachine(
    iterations=iterations,
    d_model=d_model,
    d_input=d_input,
    heads=heads,
    n_synch_out=n_synch_out,
    n_synch_action=n_synch_action,
    synapse_depth=synapse_depth,
    memory_length=memory_length,
    deep_nlms=deep_memory,
    memory_hidden_dims=memory_hidden_dims,
    do_layernorm_nlm=do_normalisation,
    backbone_type=backbone_type,
    positional_embedding_type=positional_embedding_type,
    out_dims=out_dims,
    prediction_reshaper=prediction_reshaper,
    dropout=dropout,
    dropout_nlm=dropout_nlm,
    neuron_select_type=neuron_select_type,
    n_random_pairing_self=n_random_pairing_self,
).to(device)

model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

CLASS_LIST = ['glioma', 'meningioma', 'notumor', 'pituitary']

def make_gif(input_tensor, predictions, certainties, attention_tracking, ground_truth_target, dataset_mean, dataset_std, class_labels, gif_path):

    def find_island_centers(array_2d, threshold):
        """
        Finds the center of mass of each island (connected component > threshold)
        in a 2D array, weighted by the array's values.
        Returns list of (y, x) centers and list of areas.
        """
        binary_image = array_2d > threshold
        labeled_image, num_labels = ndimage.label(binary_image)
        centers = []
        areas = []
        # Calculate center of mass for each labeled island (label 0 is background)
        for i in range(1, num_labels + 1):
            island_mask = (labeled_image == i)
            total_mass = np.sum(array_2d[island_mask])
            if total_mass > 0:
                # Get coordinates for this island
                y_coords, x_coords = np.mgrid[:array_2d.shape[0], :array_2d.shape[1]]
                # Calculate weighted average for center
                x_center = np.average(x_coords[island_mask], weights=array_2d[island_mask])
                y_center = np.average(y_coords[island_mask], weights=array_2d[island_mask])
                centers.append((round(y_center, 4), round(x_center, 4)))
                areas.append(np.sum(island_mask)) # Area is the count of pixels in the island
        return centers, areas

    interp_mode = 'nearest'
    figscale = 0.85


    # predictions: (B, Classes, Steps), attention_tracking: (Steps*B*Heads, SeqLen)
    n_steps = predictions.size(-1)

    # --- Reshape Attention ---
    # Infer feature map size from model internals (assuming B=1)
    h_feat, w_feat = model.kv_features.shape[-2:]

    n_heads = attention_tracking.shape[2] 
    # Reshape to (Steps, Heads, H_feat, W_feat) assuming B=1
    attention_tracking = attention_tracking.reshape(n_steps, n_heads, h_feat, w_feat)

    # --- Setup for Plotting ---
    step_linspace = np.linspace(0, 1, n_steps) # For step colors
    # Define color maps
    cmap_spectral = sns.color_palette("Spectral", as_cmap=True)
    cmap_attention = sns.color_palette('viridis', as_cmap=True)

    frames = [] # Store frames for GIF
    head_routes = {h: [] for h in range(n_heads)} # Store (y,x) path points per head
    head_routes[-1] = []
    route_colours_step = [] # Store colors for each step's path segments

    # --- Loop Through Each Step ---
    for step_i in range(n_steps):

        # --- Prepare Image for Display ---
        # Denormalize the input tensor for visualization
        data_img_tensor = input_tensor[0].cpu() # Get first item in batch, move to CPU
        mean_tensor = torch.tensor(dataset_mean).view(3, 1, 1)
        std_tensor = torch.tensor(dataset_std).view(3, 1, 1)
        data_img_denorm = data_img_tensor * std_tensor + mean_tensor
        # Permute to (H, W, C) and convert to numpy, clip to [0, 1]
        data_img_np = data_img_denorm.permute(1, 2, 0).detach().numpy()
        data_img_np = np.clip(data_img_np, 0, 1)
        img_h, img_w = data_img_np.shape[:2]

        # --- Process Attention & Certainty ---
        # Average attention over last few steps (from original code)
        start_step = max(0, step_i - 5)
        attention_now = attention_tracking[start_step : step_i + 1].mean(0) # Avg over steps -> (Heads, H_feat, W_feat)
        # Get certainties up to current step
        certainties_now = certainties[0, 1, :step_i+1].detach().cpu().numpy() # Assuming index 1 holds relevant certainty

        # --- Calculate Attention Paths (using bilinear interp) ---
        # Interpolate attention to image size using bilinear for center finding
        attention_interp_bilinear = F.interpolate(
            torch.from_numpy(attention_now).unsqueeze(0).float(), # Add batch dim, ensure float
            size=(img_h, img_w),
            mode=interp_mode,
            # align_corners=False
        ).squeeze(0) # Remove batch dim -> (Heads, H, W)

        # Normalize each head's map to [0, 1]
        # Deal with mean
        attn_mean = attention_interp_bilinear.mean(0)
        attn_mean_min = attn_mean.min()
        attn_mean_max = attn_mean.max()
        attn_mean = (attn_mean - attn_mean_min) / (attn_mean_max - attn_mean_min)
        centers, areas = find_island_centers(attn_mean.detach().cpu().numpy(), threshold=0.7)

        if centers: # If islands found
            largest_island_idx = np.argmax(areas)
            current_center = centers[largest_island_idx] # (y, x)
            head_routes[-1].append(current_center)
        elif head_routes[-1]: # If no center now, repeat last known center if history exists
            head_routes[-1].append(head_routes[-1][-1])


        attn_min = attention_interp_bilinear.view(n_heads, -1).min(dim=-1, keepdim=True)[0].unsqueeze(-1)
        attn_max = attention_interp_bilinear.view(n_heads, -1).max(dim=-1, keepdim=True)[0].unsqueeze(-1)
        attention_interp_bilinear = (attention_interp_bilinear - attn_min) / (attn_max - attn_min + 1e-6)

        # Store step color
        current_colour = list(cmap_spectral(step_linspace[step_i]))
        route_colours_step.append(current_colour)

        # Find island center for each head
        for head_i in range(n_heads):
            attn_head_np = attention_interp_bilinear[head_i].detach().cpu().numpy()
            # Keep threshold=0.7 based on original call
            centers, areas = find_island_centers(attn_head_np, threshold=0.7)

            if centers: # If islands found
                largest_island_idx = np.argmax(areas)
                current_center = centers[largest_island_idx] # (y, x)
                head_routes[head_i].append(current_center)
            elif head_routes[head_i]: # If no center now, repeat last known center if history exists
                    head_routes[head_i].append(head_routes[head_i][-1])
        
                

        # --- Plotting Setup ---
        mosaic = [['head_mean', 'head_mean', 'head_mean', 'head_mean', 'head_mean_overlay', 'head_mean_overlay', 'head_mean_overlay', 'head_mean_overlay'],
                    ['head_mean', 'head_mean', 'head_mean', 'head_mean', 'head_mean_overlay', 'head_mean_overlay', 'head_mean_overlay', 'head_mean_overlay'],
                    ['head_mean', 'head_mean', 'head_mean', 'head_mean', 'head_mean_overlay', 'head_mean_overlay', 'head_mean_overlay', 'head_mean_overlay'],
                    ['head_mean', 'head_mean', 'head_mean', 'head_mean', 'head_mean_overlay', 'head_mean_overlay', 'head_mean_overlay', 'head_mean_overlay'],
                ['head_0', 'head_0_overlay', 'head_1', 'head_1_overlay', 'head_2', 'head_2_overlay', 'head_3', 'head_3_overlay'],
                ['head_4', 'head_4_overlay', 'head_5', 'head_5_overlay','head_6', 'head_6_overlay', 'head_7', 'head_7_overlay'],
                ['head_8', 'head_8_overlay', 'head_9', 'head_9_overlay','head_10', 'head_10_overlay', 'head_11', 'head_11_overlay'],
                ['head_12', 'head_12_overlay', 'head_13', 'head_13_overlay','head_14', 'head_14_overlay', 'head_15', 'head_15_overlay'],
                ['probabilities', 'probabilities','probabilities', 'probabilities', 'certainty', 'certainty', 'certainty', 'certainty'],
                ]

        img_aspect = data_img_np.shape[0] / data_img_np.shape[1]
        aspect_ratio = (8 * figscale, 9 * figscale * img_aspect) # W, H
        fig, axes = plt.subplot_mosaic(mosaic, figsize=aspect_ratio)

        for ax in axes.values():
            ax.axis('off')

        # --- Plot Certainty ---
        ax_cert = axes['certainty']
        ax_cert.plot(np.arange(len(certainties_now)), certainties_now, 'k-', linewidth=figscale*1)
        # Add background color based on prediction correctness at each step
        for ii in range(len(certainties_now)):
            is_correct = predictions[0, :, ii].argmax(-1).item() == ground_truth_target # .item() for scalar tensor
            facecolor = 'limegreen' if is_correct else 'orchid'
            ax_cert.axvspan(ii, ii + 1, facecolor=facecolor, edgecolor=None, lw=0, alpha=0.3)
        # Mark the last point
        ax_cert.plot(len(certainties_now)-1, certainties_now[-1], 'k.', markersize=figscale*4)
        ax_cert.axis('off')
        ax_cert.set_ylim([0.05, 1.05])
        ax_cert.set_xlim([0, n_steps]) # Use n_steps for consistent x-axis limit

        # --- Plot Probabilities ---
        ax_prob = axes['probabilities']
        # Get probabilities for the current step
        ps = torch.softmax(predictions[0, :, step_i], -1).detach().cpu()
        k = 4 # Top k predictions
        topk_probs, topk_indices = torch.topk(ps, k, dim=0, largest=True)
        topk_indices = topk_indices.numpy()
        topk_probs = topk_probs.numpy()

        true_class_idx = ground_truth_target # Ground truth index

        # Determine bar colors (green if correct, blue otherwise - consistent with original)
        colours = ['g' if idx == true_class_idx else 'b' for idx in topk_indices]

        # Plot horizontal bars (inverted range for top-down display)
        ax_prob.barh(np.arange(k)[::-1], topk_probs, color=colours, alpha=1) # Use barh and inverted range
        ax_prob.set_xlim([0, 1])
        ax_prob.axis('off')

        # Add text labels for top classes
        for i, name_idx in enumerate(topk_indices):
            name = class_labels[name_idx] # Get name from index
            is_correct = name_idx == true_class_idx
            fg_color = 'darkgreen' if is_correct else 'crimson' # Text colors from original
            text_str = f'{name[:40]}' # Truncate long names
            # Position text on the left side of the horizontal bars
            ax_prob.text(
                0.01, # Small offset from left edge
                k - 1 - i, # Y-position corresponding to the bar
                text_str,
                #transform=ax_prob.transAxes, # Use data coordinates for Y
                verticalalignment='center',
                horizontalalignment='left',
                fontsize=8,
                color=fg_color,
                alpha=0.9, # Slightly more visible than 0.5
                path_effects=[
                    patheffects.Stroke(linewidth=2, foreground='white'), # Adjusted stroke
                    patheffects.Normal()
                ])


        # --- Plot Attention Heads & Overlays (using nearest interp) ---
        # Re-interpolate attention using nearest neighbor for visual plotting
        attention_interp_plot = F.interpolate(
            torch.from_numpy(attention_now).unsqueeze(0).float(),
            size=(img_h, img_w),
            mode=interp_mode, # 'nearest'
        ).squeeze(0)

        attn_mean = attention_interp_plot.mean(0)
        attn_mean_min = attn_mean.min()
        attn_mean_max = attn_mean.max()
        attn_mean = (attn_mean - attn_mean_min) / (attn_mean_max - attn_mean_min)


        # Normalize each head's map to [0, 1]
        attn_min_plot = attention_interp_plot.view(n_heads, -1).min(dim=-1, keepdim=True)[0].unsqueeze(-1)
        attn_max_plot = attention_interp_plot.view(n_heads, -1).max(dim=-1, keepdim=True)[0].unsqueeze(-1)
        attention_interp_plot = (attention_interp_plot - attn_min_plot) / (attn_max_plot - attn_min_plot + 1e-6)
        attention_interp_plot_np = attention_interp_plot.detach().cpu().numpy()
        


        


        for head_i in list(range(n_heads)) + [-1]:
            axname = f'head_{head_i}' if head_i != -1 else 'head_mean'
            if axname not in axes: continue # Skip if mosaic doesn't have this head

            ax = axes[axname]
            ax_overlay = axes[f'{axname}_overlay']

            # Plot attention heatmap
            this_attn = attention_interp_plot_np[head_i] if head_i != -1 else attn_mean
            img_to_plot = cmap_attention(this_attn)
            ax.imshow(img_to_plot)
            ax.axis('off')

            # Plot overlay: image + paths
            these_route_steps = head_routes[head_i]
            arrow_scale = 1.5 if head_i != -1 else 3

            if these_route_steps: # Only plot if path exists
                # Separate y and x coordinates
                y_coords, x_coords = zip(*these_route_steps)
                y_coords = np.array(y_coords)
                x_coords = np.array(x_coords)

                # Flip y-coordinates for correct plotting (imshow origin is top-left)
                # NOTE: Original flip seemed complex, simplifying to standard flip
                y_coords_flipped = img_h - 1 - y_coords

                # Show original image flipped vertically to match coordinate system
                ax_overlay.imshow(np.flipud(data_img_np), origin='lower')

                # Draw arrows for path segments
                    # Arrow size scaling from original
                for i in range(len(these_route_steps) - 1):
                    dx = x_coords[i+1] - x_coords[i]
                    dy = y_coords_flipped[i+1] - y_coords_flipped[i] # Use flipped y for delta

                    # Draw white background arrow (thicker)
                    ax_overlay.arrow(x_coords[i], y_coords_flipped[i], dx, dy,
                                        linewidth=1.6 * arrow_scale * 1.3,
                                        head_width=1.9 * arrow_scale * 1.3,
                                        head_length=1.4 * arrow_scale * 1.45,
                                        fc='white', ec='white', length_includes_head=True, alpha=1)
                    # Draw colored foreground arrow
                    ax_overlay.arrow(x_coords[i], y_coords_flipped[i], dx, dy,
                                        linewidth=1.6 * arrow_scale,
                                        head_width=1.9 * arrow_scale,
                                        head_length=1.4 * arrow_scale,
                                        fc=route_colours_step[i], ec=route_colours_step[i], # Use step color
                                        length_includes_head=True)

            else: # If no path yet, just show the image
                    ax_overlay.imshow(np.flipud(data_img_np), origin='lower')


            # Set limits and turn off axes for overlay
            ax_overlay.set_xlim([0, img_w - 1])
            ax_overlay.set_ylim([0, img_h - 1])
            ax_overlay.axis('off')
        

        # --- Finalize and Save Frame ---
        fig.tight_layout(pad=0.1) # Adjust spacing

        # Render the plot to a numpy array
        canvas = fig.canvas
        canvas.draw()
        image_numpy = np.frombuffer(canvas.buffer_rgba(), dtype='uint8')
        image_numpy = image_numpy.reshape(*reversed(canvas.get_width_height()), 4)[:,:,:3] # Get RGB

        frames.append(image_numpy) # Add to list for GIF

        

        plt.close(fig) # Close figure to free memory

    print(f"Saving GIF to {gif_path}...")
    mediapy.show_video(frames, width=400, codec="gif")
    imageio.mimsave(gif_path, frames, fps=15, loop=0) 
    pass

def inference(filename, model, output_path="output"):

    target = 0
    image = Image.open(filename).convert("RGB")
    display(image)

    dataset_mean = [
        0.23367126286029816,
        0.23367522656917572,
        0.23372678458690643
        ]
    dataset_std = [
        0.1767614781856537,
        0.1767629235982895,
        0.17677341401576996
        ]
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=dataset_mean,
            std=dataset_std
        )
    ])

    input_tensor = transform(image).unsqueeze(0).to(device)

    predictions, certainties, synchronization, pre_activations, post_activations, attention_tracking = model(input_tensor, track=True)
    gif_path = os.path.join("output", f'{filename.split('\\')[-1]}_viz.gif')
    os.makedirs("output", exist_ok=True)
    make_gif(input_tensor, predictions, certainties, attention_tracking, target, dataset_mean, dataset_std, CLASS_LIST, gif_path);

if __name__ == "__main__":
    
    filepath = r"dataset\train\pituitary\Tr-pi_0014.jpg"


    inference(filepath, model)