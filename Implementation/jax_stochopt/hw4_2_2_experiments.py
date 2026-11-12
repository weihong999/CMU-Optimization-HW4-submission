"""HW4 Exercise 2.2: reproducible Harmonic and Gaussian Langevin experiments.
Run from the jax_stochopt project root after installing the package.
"""
import argparse
import csv
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from jax_stochopt import Reduce, sum_reduce
from jax_stochopt.extensions import make_energy_langevin, make_score_langevin


def energy(params, sample, args):
    return 0.5 * jnp.sum(args['curvature'] * (params - args['mean']) ** 2) - jnp.sum(sample * (params - args['mean']))


def score(params, sample, args, temperature):
    return (-args['curvature'] * (params - args['mean']) + sample) / temperature


def sample_force(key, args):
    return args['force_scale'] * jax.random.normal(key, args['mean'].shape, dtype=args['mean'].dtype)


def gaussian_score(params, sample, args, temperature):
    del sample
    return -args['curvature'] * (params - args['mean']) / temperature


def write_csv(path, header, rows):
    with open(path, 'w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backend', choices=['cpu', 'gpu'], default='gpu')
    parser.add_argument('--chains', type=int, default=128)
    parser.add_argument('--seeds', type=int, default=10)
    parser.add_argument('--out', default='results_2_2')
    cfg = parser.parse_args()
    if cfg.chains < 10 or cfg.seeds < 2:
        parser.error('--chains must be >= 10 and --seeds must be >= 2')
    jax.config.update('jax_enable_x64', True)
    out = Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)
    dtype = jnp.float64
    device = jax.devices(cfg.backend)[0]
    print(f'Device: {device}; seeds={cfg.seeds}; Gaussian chains/seed={cfg.chains}', flush=True)

    # Harmonic：保持官方示例的全部参数和温度调度。
    steps_h = 64
    h_args = {
        'mean': jnp.asarray([0.8, -0.6, 0.3], dtype=dtype),
        'curvature': jnp.asarray([0.7, 1.3, 2.2], dtype=dtype),
        'force_scale': jnp.asarray(0.2, dtype=dtype),
        'initial_temperature': jnp.asarray(0.75, dtype=dtype),
    }
    def schedule(context):
        fraction = context.step.astype(dtype) / (steps_h - 1)
        return context.args['initial_temperature'] * (1 - 0.6 * fraction)
    h_opts = dict(steps=steps_h, learning_rate=jnp.asarray(0.04, dtype=dtype),
                  temperature=schedule, batch_size=lambda context: 4 + context.step // 4,
                  batch_buckets=(4, 8, 16, 32), record_params=True, backend=cfg.backend)
    h_energy = make_energy_langevin(energy, sample_force, **h_opts)
    h_score = make_score_langevin(score, sample_force, **h_opts)
    harmonic_rows = []
    harmonic_paths = []

    # Gaussian：保持官方示例 mean、curvature、18 steps、learning rate、temperature、initial。
    g_args = {'mean': jnp.asarray(0.7, dtype=dtype), 'curvature': jnp.asarray(1.4, dtype=dtype)}
    g_solver = make_score_langevin(gaussian_score, steps=18,
                    learning_rate=jnp.asarray(0.07, dtype=dtype),
                    temperature=jnp.asarray(0.7, dtype=dtype),
                    record_params=True, backend=cfg.backend)
    gaussian_means = []
    gaussian_variances = []
    gaussian_paths_0 = None
    gaussian_rows = []
    with jax.default_device(device):
        for seed in range(cfg.seeds):
            base = jax.random.key(seed, impl='threefry2x32')
            h_key, g_key = jax.random.split(base, 2)
            er = h_energy.run(jnp.zeros_like(h_args['mean']), h_key, h_args)
            sr = h_score.run(jnp.zeros_like(h_args['mean']), h_key, h_args)
            ep = np.asarray(jax.device_get(er.history.params))
            sp = np.asarray(jax.device_get(sr.history.params))
            diff = float(np.max(np.abs(ep - sp)))
            harmonic_rows.append([seed, diff])
            if seed == 0:
                harmonic_paths = [ep, sp]
            print(f'Seed {seed}: Harmonic max difference = {diff:.3e}', flush=True)

            keys = jax.random.split(g_key, cfg.chains)
            # 每个 seed 包含相互独立的 Gaussian Chain，保留完整轨迹。
            trajectories = jax.jit(jax.vmap(
                lambda k: g_solver.run(jnp.asarray(-1.8, dtype=dtype), k, g_args).history.params
            ))(keys)
            arr = np.asarray(jax.device_get(trajectories))  # (chains, steps)
            gaussian_means.append(arr.mean(axis=0))
            gaussian_variances.append(arr.var(axis=0, ddof=1))
            gaussian_rows.append([seed, float(arr[:, -1].mean()), float(arr[:, -1].var(ddof=1))])
            if seed == 0:
                gaussian_paths_0 = arr[:10].copy()
            print(f'        Gaussian final mean = {gaussian_rows[-1][1]:.6f}, variance = {gaussian_rows[-1][2]:.6f}', flush=True)

    write_csv(out / 'harmonic_seed_errors.csv', ['seed', 'max_abs_path_difference'], harmonic_rows)
    write_csv(out / 'gaussian_seed_final_moments.csv', ['seed', 'final_mean', 'final_variance'], gaussian_rows)
    max_diff = max(row[1] for row in harmonic_rows)
    hm = np.arange(1, steps_h + 1)
    fig, axs = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
    for dim, ax in enumerate(axs):
        ax.plot(hm, harmonic_paths[0][:, dim], label='Energy', lw=2)
        ax.plot(hm, harmonic_paths[1][:, dim], '--', label='Score', lw=1.3)
        ax.set_ylabel(f'$x_{{{dim+1}}}$')
        ax.grid(alpha=0.3)
    axs[0].legend()
    axs[-1].set_xlabel('Iteration')
    fig.suptitle('Harmonic: Energy vs Score (seed 0)')
    fig.tight_layout()
    fig.savefig(out / 'harmonic_energy_vs_score.png', dpi=200)
    plt.close(fig)

    gm = np.arange(1, 19)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, path in enumerate(gaussian_paths_0):
        ax.plot(gm, path, lw=1, alpha=0.8, label=f'Chain {i + 1}')
    ax.axhline(0.7, color='black', ls='--', lw=1.5, label='Target mean 0.7')
    ax.set(xlabel='Iteration', ylabel='State', title='Gaussian: 10 representative chains (seed 0)')
    ax.legend(ncol=3, fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / 'gaussian_10_chains.png', dpi=200)
    plt.close(fig)

    means = np.stack(gaussian_means)
    variances = np.stack(gaussian_variances)
    avg = means.mean(axis=0)
    sd = means.std(axis=0, ddof=1)
    write_csv(out / 'gaussian_mean_sd_by_iteration.csv',
              ['iteration', 'mean_across_seeds', 'sd_across_seed_means'],
              zip(gm, avg, sd))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.errorbar(gm, avg, yerr=sd, fmt='o-', ms=3, capsize=2,
                label=f'Mean ± SD across {cfg.seeds} seeds')
    ax.axhline(0.7, color='black', ls='--', lw=1.5, label='Target mean 0.7')
    ax.set(xlabel='Iteration', ylabel='Ensemble mean', title='Gaussian: ensemble means across seeds')
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / 'gaussian_mean_sd.png', dpi=200)
    plt.close(fig)

    report = {
        'backend': str(device), 'seeds': cfg.seeds, 'chains_per_seed': cfg.chains,
        'harmonic_max_difference_across_seeds': max_diff,
        'gaussian_target_mean': 0.7, 'gaussian_target_variance': 0.5,
        'gaussian_final_mean_across_seed_means': float(avg[-1]),
        'gaussian_final_sd_across_seed_means': float(sd[-1]),
        'gaussian_final_mean_of_within_seed_variances': float(variances[:, -1].mean()),
        'note': '18-step unadjusted Langevin is not an exact equilibrium sampler.'
    }
    (out / 'summary.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
    print(f'Figures/CSVs saved under {out.resolve()}', flush=True)


if __name__ == '__main__':
    main()
