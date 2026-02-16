# faber-nn-nifs

[![Buy Me A Coffee](https://img.shields.io/badge/Buy%20Me%20A%20Coffee-support-yellow.svg)](https://buymeacoffee.com/rlefever)

High-performance Rust NIFs (Native Implemented Functions) for Faber Neuroevolution.

This is an **enterprise-only** package that provides 10-15x performance improvements over pure Erlang fallbacks for compute-intensive neuroevolution operations.

## Installation

Add to your `rebar.config`:

```erlang
{deps, [
    {faber_tweann, "~> 1.0"},
    {faber_nn_nifs, {git, "git@github.com:rgfaber/faber-nn-nifs.git", {tag, "v1.0.0"}}}
]}.
```

The NIFs are automatically detected and used by `faber_tweann`. No code changes required.

## Requirements

- Erlang/OTP 26+
- Rust 1.70+ (for building NIFs)
- Cargo (Rust package manager)

## Building

```bash
rebar3 compile
```

This automatically:
1. Compiles the Rust NIF library via Cargo
2. Copies the compiled `.so`/`.dylib`/`.dll` to `priv/`
3. Compiles the Erlang wrapper module

## Verification

```erlang
1> faber_nn_nifs:is_loaded().
true

2> faber_nn_nifs:random_weights(5).
[0.123, -0.456, 0.789, -0.234, 0.567]
```

## API Categories

### Network Evaluation
- `compile_network/3` - Compile network topology for fast evaluation
- `evaluate/2` - Forward propagation through compiled network
- `evaluate_batch/2` - Batch evaluation for multiple inputs
- `compatibility_distance/5` - NEAT speciation distance
- `benchmark_evaluate/3` - Performance benchmarking

### Signal Aggregation
- `dot_product_flat/3` - Fast weighted sum with bias
- `dot_product_batch/1` - Batch weighted sums
- `dot_product_preflattened/3` - Pre-optimized dot product
- `flatten_weights/1` - Weight structure optimization

### LTC/CfC (Liquid Time-Constant Networks)
- `evaluate_cfc/4` - Closed-form continuous-time evaluation
- `evaluate_cfc_with_weights/6` - CfC with custom weights
- `evaluate_ode/5` - ODE-based LTC evaluation
- `evaluate_ode_with_weights/7` - ODE with custom weights
- `evaluate_cfc_batch/4` - Batch CfC for time series

### Novelty Search
- `euclidean_distance/2` - Vector distance
- `euclidean_distance_batch/2` - Batch distances
- `knn_novelty/4` - K-nearest neighbor novelty score
- `knn_novelty_batch/3` - Batch novelty computation

### Statistics
- `fitness_stats/1` - Single-pass min/max/mean/variance/stddev/sum
- `weighted_moving_average/2` - WMA computation
- `shannon_entropy/1` - Entropy calculation
- `histogram/4` - Histogram binning

### Selection
- `build_cumulative_fitness/1` - Roulette wheel setup
- `roulette_select/3` - Single roulette selection
- `roulette_select_batch/3` - Batch selection
- `tournament_select/2` - Tournament selection

### Meta-Controller
- `z_score/3` - Z-score normalization
- `compute_reward_component/2` - Reward signal computation
- `compute_weighted_reward/1` - Multi-component rewards

### Evolutionary Genetics
- `mutate_weights/4` - Gaussian weight mutation
- `mutate_weights_seeded/5` - Reproducible mutation
- `mutate_weights_batch/1` - Batch mutation with per-genome params
- `mutate_weights_batch_uniform/4` - Batch with uniform params
- `random_weights/1` - Generate random weights [-1, 1]
- `random_weights_seeded/2` - Seeded random weights
- `random_weights_gaussian/3` - Gaussian distributed weights
- `random_weights_batch/1` - Batch weight generation
- `weight_distance_l1/2` - L1 (Manhattan) distance
- `weight_distance_l2/2` - L2 (Euclidean) distance
- `weight_distance_batch/3` - Batch distance computation

### Layer-specific Mutation
- `mutate_weights_layered/6` - Different rates for reservoir vs readout layers
- `compute_layer_weight_counts/1` - Weight counts per layer from topology

### SIMD Batch Activations
- `tanh_batch/1`, `sigmoid_batch/1`, `relu_batch/1`, `softmax_batch/1`
- `activation_batch/2` - Specified activation function

### Plasticity
- `hebbian_update_batch/4` - Hebbian weight update
- `modulated_hebbian_batch/5` - Reward-modulated Hebbian
- `stdp_update/5` - Spike-Timing Dependent Plasticity
- `oja_update_batch/4` - Normalized Hebbian (Oja's rule)

### Time Series LTC/CfC
- `evaluate_cfc_sequence/5` - CfC over input sequence
- `evaluate_cfc_parallel/4` - Parallel CfC neurons
- `ltc_state_batch/4` - Batch LTC state update

### Population Diversity
- `population_diversity/1` - Diversity metrics
- `weight_covariance_matrix/1` - Covariance matrix (for CMA-ES)
- `pairwise_distances_batch/2` - All pairwise distances

### NEAT Crossover
- `neat_crossover/4` - NEAT-style crossover
- `align_genes_by_innovation/2` - Gene alignment
- `count_excess_disjoint/2` - Excess/disjoint gene counts

### Speciation Clustering
- `assign_species_batch/3` - Batch species assignment
- `find_representative/2` - Find species representative
- `kmeans_cluster/3` - K-means clustering

### Matrix Operations
- `matmul_add_bias/3` - Matrix multiply with bias
- `layer_forward/4` - Single layer forward pass
- `multi_layer_forward/3` - Multi-layer forward pass

## Performance

Typical speedups over pure Erlang:

| Operation | Speedup |
|-----------|---------|
| Network evaluate | ~10x |
| Batch mutation | ~13x |
| KNN novelty | ~12x |
| Fitness stats | ~12x |
| Weight distance | ~15x |

## How It Works

The `tweann_nif` module in `faber_tweann` automatically detects this package:

```erlang
%% Priority order:
%% 1. faber_nn_nifs (enterprise - this package)
%% 2. Bundled NIF in faber_tweann
%% 3. Pure Erlang fallback

detect_impl_module() ->
    case code:which(faber_nn_nifs) of
        non_existing -> tweann_nif_fallback;
        _ ->
            case faber_nn_nifs:is_loaded() of
                true -> faber_nn_nifs;
                false -> tweann_nif_fallback
            end
    end.
```

## Testing

```bash
rebar3 eunit
```

## Related

- [faber_tweann](https://hex.pm/packages/faber_tweann) - Community edition on hex.pm
- [faber_neuroevolution](https://hex.pm/packages/faber_neuroevolution) - Population-based evolution
- [faber-ecosystem](https://github.com/rgfaber/faber-ecosystem) - Ecosystem documentation

## License

Apache-2.0
