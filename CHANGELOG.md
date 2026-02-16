# Changelog

All notable changes to faber-nn-nifs will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-02-16

### Stable Release

First stable release. All APIs are considered stable.

### Features

- **Network Evaluation**: `compile_network/3`, `evaluate/2`, `evaluate_batch/2`
- **Signal Aggregation**: `dot_product_flat/3`, `dot_product_batch/1`
- **LTC/CfC Networks**: `evaluate_cfc/4`, `evaluate_ode/5`, `evaluate_cfc_batch/4`, `evaluate_cfc_sequence/5`
- **Novelty Search**: `euclidean_distance/2`, `knn_novelty/4`, `knn_novelty_batch/3`
- **Statistics**: `fitness_stats/1`, `weighted_moving_average/2`, `shannon_entropy/1`
- **Selection**: `tournament_select/2`, `roulette_select/3`, `build_cumulative_fitness/1`
- **Evolutionary Genetics**: `mutate_weights/4`, `random_weights/1`, `neat_crossover/4`
- **Layer-specific Mutation**: `mutate_weights_layered/6`, `compute_layer_weight_counts/1`
- **SIMD Batch Activations**: `tanh_batch/1`, `sigmoid_batch/1`, `relu_batch/1`, `softmax_batch/1`
- **Plasticity**: `hebbian_update_batch/4`, `modulated_hebbian_batch/5`, `stdp_update/5`, `oja_update_batch/4`
- **Population Diversity**: `population_diversity/1`, `pairwise_distances_batch/2`
- **Speciation**: `assign_species_batch/3`, `kmeans_cluster/3`
- **Matrix Operations**: `matmul_add_bias/3`, `layer_forward/4`, `multi_layer_forward/3`

### Changed

- License corrected to Apache-2.0 (was incorrectly MIT in LICENSE file)

---

## [0.1.0] - 2026-02-14

### Changed
- Renamed from `macula-nn-nifs` to `faber-nn-nifs` under `rgfaber` organization
- Reset version to 0.1.0 for fresh start under new name

---

[1.0.0]: https://github.com/rgfaber/faber-nn-nifs/compare/v0.1.0...v1.0.0
[0.1.0]: https://github.com/rgfaber/faber-nn-nifs/releases/tag/v0.1.0
