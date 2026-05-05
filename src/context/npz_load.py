import numpy as np

d = np.load("heatmap_snapshot_20260428_203710.npz", allow_pickle=False)

# raw gaussian info (pulled straight from self.render_kwargs["pc"])
xyz          = d["xyz"]           # (N, 3)
features_dc  = d["features_dc"]  # (N, 1, 3) base SH colour
features_rest= d["features_rest"]# (N, R, 3) higher-order SH
scaling      = d["scaling"]      # (N, 3)
rotation     = d["rotation"]     # (N, 4)
opacity      = d["opacity"]      # (N, 1)

active_sh_degree = int(d["active_sh_degree"])
max_sh_degree    = int(d["max_sh_degree"])
spatial_lr_scale = float(d["spatial_lr_scale"])

N = xyz.shape[0]

# embeddings
embeddings   = d["embeddings"]   # (N, D)   NaN row = gaussian was never embedded
has_embedding = ~np.isnan(embeddings[:, 0])   # (N,) bool mask

# backprojected heatmap values, per gaussian
heatmap_values = d["heatmap_values"]           # (N,)  float32, NaN = not visible
has_heatmap    = ~np.isnan(heatmap_values)     # (N,) bool mask

# subject + context selections (indices into the N gaussians)
subject_idx = d["subject_indices"]   # (M,) int64 subject gaussians
context_idx = d["context_indices"]   # (K,) int64 context gaussians

subject_mask = np.zeros(N, dtype=bool)
context_mask = np.zeros(N, dtype=bool)
subject_mask[subject_idx] = True
context_mask[context_idx] = True

print(f"N={N}  embedded={has_embedding.sum()}  "
      f"heatmap={has_heatmap.sum()}  "
      f"subject={subject_mask.sum()}  context={context_mask.sum()}")

# Returns (subject_centroid - context_centroid) as a (3,) vector.
def compute_subject_context_offset(data):
    xyz = data["xyz"]
    subject_idx = data["subject_indices"]
    context_idx = data["context_indices"]
    offset = xyz[subject_idx].mean(axis=0) - xyz[context_idx].mean(axis=0)
    print(f"subject centroid : {xyz[subject_idx].mean(axis=0)}")
    print(f"context centroid : {xyz[context_idx].mean(axis=0)}")
    print(f"offset (sub-ctx) : {offset}")
    return offset

compute_subject_context_offset(d)