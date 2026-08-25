# PATCH for Backend/app/ml/model_runner.py
# ==========================================
# Goal: Compute and save per-group feature statistics + similarity matrix
# so the dashboard can display the deep-learning features visually.
#
# Apply by running this Python script from the Backend/ directory.

import re

PATH = "app/ml/model_runner.py"
src = open(PATH).read()

# ── 1. Replace _compute_reid_groups to compute & embed feature stats ──────
old_block = '''        reid_groups = []
        for gid, indices in enumerate(groups):
            vid = f"VH-{gid + 1:03d}"
            timestamps = [detections[i].timestamp for i in indices]

            if len(indices) > 1:
                pairs = [float(sim_matrix[indices[a], indices[b]])
                         for a in range(len(indices))
                         for b in range(a + 1, len(indices))]
                best_score = float(np.mean(pairs))
            else:
                best_score = 1.0

            # ── consensus plate number for this group ─────────────────────
            plates = [detections[i].plate_number for i in indices if detections[i].plate_number]
            consensus_plate = None
            if plates:
                from collections import Counter
                most_common = Counter(plates).most_common(1)
                consensus_plate = most_common[0][0]

            reid_groups.append({
                "vehicle_id": vid,
                "detection_indices": indices,
                "detection_count": len(indices),
                "best_score": round(best_score, 3),
                "first_seen": round(min(timestamps), 2),
                "last_seen": round(max(timestamps), 2),
                "plate_number": consensus_plate,
            })'''

new_block = '''        reid_groups = []
        for gid, indices in enumerate(groups):
            vid = f"VH-{gid + 1:03d}"
            timestamps = [detections[i].timestamp for i in indices]

            if len(indices) > 1:
                pairs = [float(sim_matrix[indices[a], indices[b]])
                         for a in range(len(indices))
                         for b in range(a + 1, len(indices))]
                best_score = float(np.mean(pairs))
            else:
                best_score = 1.0

            # ── consensus plate number for this group ─────────────────────
            plates = [detections[i].plate_number for i in indices if detections[i].plate_number]
            consensus_plate = None
            if plates:
                from collections import Counter
                most_common = Counter(plates).most_common(1)
                consensus_plate = most_common[0][0]

            # ── FEATURE STATISTICS (deep-learning embedding profile) ──────
            # Average the embeddings of all detections in this group → group centroid
            group_embeddings = np.stack([embeddings[i] for i in indices])
            centroid = group_embeddings.mean(axis=0)
            # Re-normalize centroid (since average of unit vectors is not unit)
            centroid = centroid / (np.linalg.norm(centroid) + 1e-8)

            # Top activations (for the "key features" display)
            abs_centroid = np.abs(centroid)
            top_k = 8
            top_indices = np.argsort(abs_centroid)[-top_k:][::-1]
            top_activations = [
                {"dim": int(idx), "value": float(centroid[idx])}
                for idx in top_indices
            ]

            # Histogram-friendly summary: compress 512 dims to 64 bins
            # by averaging chunks (so the heatmap strip is renderable inline)
            n_bins = 64
            bin_size = max(1, len(centroid) // n_bins)
            heatmap_strip = [
                float(np.mean(centroid[k * bin_size:(k + 1) * bin_size]))
                for k in range(n_bins)
            ]

            reid_groups.append({
                "vehicle_id": vid,
                "detection_indices": indices,
                "detection_count": len(indices),
                "best_score": round(best_score, 3),
                "first_seen": round(min(timestamps), 2),
                "last_seen": round(max(timestamps), 2),
                "plate_number": consensus_plate,
                # ── Feature display fields ──
                "feature_stats": {
                    "embedding_dim": int(len(centroid)),
                    "magnitude": float(np.linalg.norm(centroid)),
                    "mean": float(centroid.mean()),
                    "std": float(centroid.std()),
                    "min": float(centroid.min()),
                    "max": float(centroid.max()),
                    "top_activations": top_activations,
                    "heatmap_strip": [round(v, 4) for v in heatmap_strip],
                    "centroid_preview": [round(float(v), 4) for v in centroid[:32].tolist()],
                },
            })

        # ── Build group-vs-group similarity matrix for the UI ─────────────
        if len(reid_groups) >= 2:
            centroids = []
            for g in reid_groups:
                grp_embs = np.stack([embeddings[i] for i in g["detection_indices"]])
                c = grp_embs.mean(axis=0)
                c = c / (np.linalg.norm(c) + 1e-8)
                centroids.append(c)
            cent_matrix = np.stack(centroids)
            inter_sim = cent_matrix @ cent_matrix.T
            for i, g in enumerate(reid_groups):
                g["similarity_to_others"] = [
                    {
                        "vehicle_id": reid_groups[j]["vehicle_id"],
                        "similarity": round(float(inter_sim[i, j]), 3),
                    }
                    for j in range(len(reid_groups)) if j != i
                ]
        else:
            for g in reid_groups:
                g["similarity_to_others"] = []'''

if old_block in src:
    src = src.replace(old_block, new_block)
    open(PATH, "w").write(src)
    print("✓ model_runner.py patched with feature_stats + similarity_to_others")
else:
    print("✗ Pattern not found — paste the file and we'll do it manually")