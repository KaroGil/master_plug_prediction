"""
Helper method to identify and analyse datasets where any model scores below a certain F1 threshold.
"""

import numpy as np

def analyse_low_performing_datasets(dataset_ids, f1_scores_dict, X_y_list, threshold=0.8):
    """
    Identifies and analyses datasets where any model scores
    below the threshold F1.
    """

    low_ids = []
    for i, ds_id in enumerate(dataset_ids):
        scores = {name: scores[i] 
                  for name, scores in f1_scores_dict.items()}
        if any(s < threshold for s in scores.values()):
            low_ids.append((i, ds_id, scores))

    if not low_ids:
        print(f"All datasets performed above the threshold of {threshold}.")
        return []
    else: 
        print(f"\n{'─'*55}")
        print(f"  Datasets with F1 < {threshold} for any model")
        print(f"{'─'*55}")

    for iloc, ds_id, scores in low_ids:
        _, y = X_y_list[iloc]
        y = np.array(y)

        n_samples   = len(y)
        n_plug      = (y == 1).sum()
        n_no_plug   = (y == 0).sum()
        plug_ratio  = n_plug / n_samples

        print(f"\n  Dataset {ds_id}")
        print(f"  {'─'*40}")
        print(f"  Samples      : {n_samples}")
        print(f"  Plug (1)     : {n_plug} ({plug_ratio*100:.1f}%)")
        print(f"  No Plug (0)  : {n_no_plug} ({(1-plug_ratio)*100:.1f}%)")
        for model_name, score in scores.items():
            flag = " ← below threshold" if score < threshold else ""
            print(f"  {model_name:<20}: F1 = {score:.4f}{flag}")

    return [ds_id for _, ds_id, _ in low_ids]