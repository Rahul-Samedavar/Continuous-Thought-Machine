import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import patheffects
from scipy import ndimage
import imageio


def make_gif(
    input_tensor,
    predictions,
    certainties,
    attention_tracking,
    ground_truth_target,
    dataset_mean,
    dataset_std,
    class_labels,
    gif_path,
):
    def find_island_centers(array_2d, threshold):
        binary = array_2d > threshold
        labeled, num = ndimage.label(binary)

        centers, areas = [], []
        for i in range(1, num + 1):
            mask = labeled == i
            if mask.sum() == 0:
                continue

            y, x = np.mgrid[:array_2d.shape[0], :array_2d.shape[1]]
            weights = array_2d[mask]
            centers.append((
                float(np.average(y[mask], weights=weights)),
                float(np.average(x[mask], weights=weights)),
            ))
            areas.append(mask.sum())

        return centers, areas

    interp_mode = "nearest"
    figscale = 0.85
    n_steps = predictions.size(-1)

    # Infer shape
    h_feat, w_feat = attention_tracking.shape[-2:]
    n_heads = attention_tracking.shape[2]

    attention_tracking = attention_tracking.reshape(
        n_steps, n_heads, h_feat, w_feat
    )

    cmap_attention = sns.color_palette("viridis", as_cmap=True)
    cmap_steps = sns.color_palette("Spectral", as_cmap=True)

    frames = []
    head_routes = {h: [] for h in range(n_heads)}
    head_routes[-1] = []
    step_colors = []

    for step in range(n_steps):
        # ---------------- Image ----------------
        img = input_tensor[0].cpu()
        mean = torch.tensor(dataset_mean).view(3, 1, 1)
        std = torch.tensor(dataset_std).view(3, 1, 1)
        img = (img * std + mean).permute(1, 2, 0).numpy()
        img = np.clip(img, 0, 1)

        H, W = img.shape[:2]

        # ---------------- Attention ----------------
        attn = attention_tracking[max(0, step - 5): step + 1].mean(0)

        attn_img = F.interpolate(
            torch.from_numpy(attn).unsqueeze(0).float(),
            size=(H, W),
            mode=interp_mode,
        ).squeeze(0)

        # Mean attention
        attn_mean = attn_img.mean(0)
        attn_mean = (attn_mean - attn_mean.min()) / (attn_mean.max() - attn_mean.min() + 1e-6)

        centers, areas = find_island_centers(attn_mean.numpy(), 0.7)
        if centers:
            head_routes[-1].append(centers[np.argmax(areas)])
        elif head_routes[-1]:
            head_routes[-1].append(head_routes[-1][-1])

        for h in range(n_heads):
            ah = attn_img[h]
            ah = (ah - ah.min()) / (ah.max() - ah.min() + 1e-6)

            centers, areas = find_island_centers(ah.numpy(), 0.7)
            if centers:
                head_routes[h].append(centers[np.argmax(areas)])
            elif head_routes[h]:
                head_routes[h].append(head_routes[h][-1])

        step_colors.append(cmap_steps(step / n_steps))

        # ---------------- Layout ----------------
        mosaic = [
            ["mean", "mean", "mean", "mean", "mean_o", "mean_o", "mean_o", "mean_o"],
            ["mean", "mean", "mean", "mean", "mean_o", "mean_o", "mean_o", "mean_o"],
            ["prob", "prob", "prob", "prob", "cert", "cert", "cert", "cert"],
        ]

        fig, axes = plt.subplot_mosaic(mosaic, figsize=(10 * figscale, 6 * figscale))
        for ax in axes.values():
            ax.axis("off")

        # ---------------- Certainty ----------------
        cert = certainties[0, 1, : step + 1].cpu().numpy()
        ax = axes["cert"]
        ax.plot(cert, "k-", lw=1)
        ax.set_ylim(0, 1.05)

        for i in range(len(cert)):
            correct = predictions[0, :, i].argmax().item() == ground_truth_target
            ax.axvspan(i, i + 1, color="limegreen" if correct else "orchid", alpha=0.3)

        # ---------------- Probabilities ----------------
        probs = torch.softmax(predictions[0, :, step], -1).cpu()
        topk = torch.topk(probs, 4)

        ax = axes["prob"]
        for i, (idx, val) in enumerate(zip(topk.indices, topk.values)):
            correct = idx.item() == ground_truth_target
            ax.barh(3 - i, val.item(), color="g" if correct else "b")
            ax.text(
                0.01,
                3 - i,
                class_labels[idx],
                va="center",
                fontsize=8,
                color="darkgreen" if correct else "crimson",
                path_effects=[
                    patheffects.Stroke(linewidth=2, foreground="white"),
                    patheffects.Normal(),
                ],
            )
        ax.set_xlim(0, 1)

        # ---------------- Attention mean + path ----------------
        axes["mean"].imshow(cmap_attention(attn_mean))
        axes["mean_o"].imshow(np.flipud(img), origin="lower")

        path = head_routes[-1]
        if len(path) > 1:
            y, x = zip(*path)
            y = np.array(y)
            x = np.array(x)
            y = H - 1 - y

            for i in range(len(x) - 1):
                axes["mean_o"].arrow(
                    x[i],
                    y[i],
                    x[i + 1] - x[i],
                    y[i + 1] - y[i],
                    color=step_colors[i],
                    linewidth=2,
                    head_width=3,
                    length_includes_head=True,
                )

        # ---------------- Render ----------------
        fig.tight_layout(pad=0.1)
        fig.canvas.draw()

        frame = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        frames.append(frame[:, :, :3])

        plt.close(fig)

    os.makedirs(os.path.dirname(gif_path), exist_ok=True)
    imageio.mimsave(gif_path, frames, fps=15, loop=0)
