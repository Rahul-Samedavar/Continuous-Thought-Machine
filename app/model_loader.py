import torch
from models.ctm import ContinuousThoughtMachine

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

def load_model(checkpoint_path: str):
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)

    model = ContinuousThoughtMachine(
        iterations=75,
        d_model=512,
        d_input=128,
        heads=4,
        n_synch_out=512,
        n_synch_action=512,
        synapse_depth=4,
        memory_length=25,
        deep_nlms=True,
        memory_hidden_dims=4,
        do_layernorm_nlm=False,
        backbone_type="resnet18-4",
        positional_embedding_type="none",
        out_dims=4,
        prediction_reshaper=[-1],
        dropout=0.0,
        dropout_nlm=None,
        neuron_select_type="random-pairing",
        n_random_pairing_self=0,
    ).to(DEVICE)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model
