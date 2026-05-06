import numpy as np
from script.helper_methods.data_visualization import plot_test_f1_vs_horizon

#score
# Horizon: 1 samples
# Model chosen RF
# RF: 0.8283892847483973
# XGB: 0.8271948777403167
# Test: 0.9826639008353253
# Horizon: 10 samples
# Model chosen RF
# RF: 0.8360082880949147
# XGB: 0.8164881050179208
# Test: 0.9860107392454315
# Horizon: 50 samples
# Model chosen XGB
# RF: 0.8137908728214578
# XGB: 0.8192883239446836
# Test: 0.9927024604198837
# Horizon: 100 samples
# Model chosen XGB
# RF: 0.7930648142451427
# XGB: 0.7997662343986559
# Test: 0.9933143227066209
# Horizon: 500 samples
# Model chosen RF
# RF: 0.7936797209574435
# XGB: 0.7819042766344303
# Test: 0.8840874192591732

# === SUMMARY OF SCORES ===
# Horizon: 5 samples
# Model chosen RF
# RF: 0.8298750326086988
# XGB: 0.8223089002270525
# Test: 0.9854071951409029
# Horizon: 10 samples
# Model chosen RF
# RF: 0.8217399814688475
# XGB: 0.8137604740624265
# Test: 0.9851022732677731
# Horizon: 15 samples
# Model chosen RF
# RF: 0.8208845540339196
# XGB: 0.8138284982425511
# Test: 0.9875356400872458
# Horizon: 25 samples
# Model chosen RF
# RF: 0.8184790949970103
# XGB: 0.815891391812486
# Test: 0.9899698373462416
# Horizon: 50 samples
# Model chosen RF
# RF: 0.8232102161968246
# XGB: 0.8189206167729272
# Test: 0.9942248707482748

# === SUMMARY OF SCORES ===
# Horizon: 5 samples
# Model chosen RF
# RF: 0.8298750326086988
# XGB: 0.8223089002270525
# Test: 0.9854071951409029
# Horizon: 10 samples
# Model chosen RF
# RF: 0.8217399814688475
# XGB: 0.8137604740624265
# Test: 0.9851022732677731
# Horizon: 15 samples
# Model chosen RF
# RF: 0.8208845540339196
# XGB: 0.8138284982425511
# Test: 0.9875356400872458
# Horizon: 25 samples
# Model chosen RF
# RF: 0.8184790949970103
# XGB: 0.815891391812486
# Test: 0.9899698373462416
# Horizon: 50 samples
# Model chosen RF
# RF: 0.8232102161968246
# XGB: 0.8189206167729272
# Test: 0.9942248707482748

horizons1 = [5, 10, 15, 25, 50]  # seconds
horizons2 = [1, 10, 50, 100, 500]
horizons = horizons1 + horizons2
horizons = sorted(horizons)
# scores = {
#     1: (0.8283892847483973, 0.8271948777403167, 0.9826639008353253),
#     5: (0.8298750326086988, 0.8223089002270525, 0.9854071951409029),
#     10: (0.8360082880949147, 0.8164881050179208, 0.9860107392454315),
#     15: (0.8208845540339196, 0.8138284982425511, 0.9875356400872458),
#     25: (0.8184790949970103, 0.815891391812486, 0.9899698373462416),
#     50: (0.8232102161968246, 0.8189206167729272, 0.9942248707482748),
#     100: (0.7930648142451427, 0.7997662343986559, 0.9933143227066209),
#     500: (0.7936797209574435, 0.7819042766344303, 0.8840874192591732),
# }
# plot_test_f1_vs_horizon(horizons, [scores[h][0] for h in horizons], test_or_val="RF")
# plot_test_f1_vs_horizon(horizons, [scores[h][1] for h in horizons], test_or_val="XGB")
# plot_test_f1_vs_horizon(horizons, [scores[h][2] for h in horizons], test_or_val="T")

score1 = {
    5: (0.8298750326086988, 0.8223089002270525, 0.9854071951409029),
    10: (0.8217399814688475, 0.8137604740624265, 0.9851022732677731),
    15: (0.8208845540339196, 0.8138284982425511, 0.9875356400872458),
    25: (0.8184790949970103, 0.815891391812486, 0.9899698373462416),
    50: (0.8232102161968246, 0.8189206167729272, 0.9942248707482748),
}
score2 = {
    1: (0.8283892847483973, 0.8271948777403167, 0.9826639008353253),
    10: (0.8360082880949147, 0.8164881050179208, 0.9860107392454315),
    50: (0.8137908728214578, 0.8192883239446836, 0.9927024604198837),
    100: (0.7930648142451427, 0.7997662343986559, 0.9933143227066209),
    500: (0.7936797209574435, 0.7819042766344303, 0.8840874192591732),
}

def find_best_horizon(scores):
    horizons = list(scores.keys())
    best_val_scores = [max(x[0], x[1]) for x in scores.values()]
    best_horizon_idx = np.argmax(best_val_scores)
    best_horizon = horizons[best_horizon_idx]
    best_score = scores[best_horizon]
    best_model_name = "RF" if best_score[0] > best_score[1] else "XGB"
    print(f"\nBest horizon: {best_horizon} samples")
    print(f"Best model: {best_model_name}")
    print(f"Validation F1: {best_val_scores[best_horizon_idx]:.4f}")
    print(f"Test F1: {best_score[2]:.4f}")
    return best_horizon, best_model_name, best_val_scores[best_horizon_idx], best_score[2]

best_horizon1, best_model1, best_val1, test1 = find_best_horizon(score1)
best_horizon2, best_model2, best_val2, test2 = find_best_horizon(score2)

print("\n === COMPARISON OF BEST HORIZONS ===")
print(f"Best horizon: {best_horizon1} samples" if best_val1 > best_val2 else f"Best horizon: {best_horizon2} samples")
print(f"Best model: {best_model1}" if best_val1 > best_val2 else f"Best model: {best_model2}")
print(f"Best validation F1: {best_val1:.4f}" if best_val1 > best_val2 else f"Best validation F1: {best_val2:.4f}")
print(f"Test F1 for best horizon: {test1:.4f}" if best_val1 > best_val2 else f"Test F1 for best horizon: {test2:.4f}")
