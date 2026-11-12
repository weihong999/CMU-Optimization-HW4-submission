"""Exercise 2.1 SGD hyperparameter sweep; run in the jax_stochopt project root.

Uses the supplied Solver, samplers, objectives and simplex projection.
Does not edit any course files. Produces per-seed histories, mean +/- SD,
iteration- and sample-axis figures, and best-configuration summaries.
"""
import argparse
import csv
import json
import os
from pathlib import Path

os.environ.setdefault('MPLBACKEND', 'Agg')
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from jax_stochopt import SGD, Reduce, make_solver, sum_reduce
from jax_stochopt.projections import simplex


def sampler(key, args):
    return jax.random.normal(key, args['target'].shape, dtype=args['target'].dtype)


def quadratic_loss(x, z, args):
    d = x - args['target']
    return 0.5 * jnp.sum(args['curvature'] * d**2) - args['noise_scale'] * jnp.sum(z*d)


def congestion_energy(x, z, args):
    del z, args
    return 0.5 * jnp.sum(x**2)


def random_benefit(x, z, args):
    return -jnp.sum((args['target'] + args['noise_scale']*z) * x)


def run_one(problem, lr, batch, seed, steps, dtype, backend):
    if problem == 'quadratic':
        args = {
            'target': jnp.asarray([0.8, -0.6, 0.3, 1.1], dtype=dtype),
            'curvature': jnp.asarray([0.7, 1.3, 2.2, 4.0], dtype=dtype),
            'noise_scale': jnp.asarray(0.35, dtype=dtype),
        }
        x0 = jnp.zeros_like(args['target'])
        objective = quadratic_loss
        optimizer = SGD(momentum=0.6)  # same as course example
        projection = None
    else:
        args = {
            'target': jnp.asarray([0.70, 0.45, 0.15, -0.30], dtype=dtype),
            'noise_scale': jnp.asarray(0.15, dtype=dtype),
        }
        x0 = jnp.full((4,), 0.25, dtype=dtype)
        objective = Reduce(sum_reduce, (congestion_energy, random_benefit))
        optimizer = SGD()  # same as course example
        projection = simplex(total=1.0)

    solver = make_solver(
        objective, sampler, optimizer=optimizer,
        learning_rate=float(lr), batch_size=int(batch),
        max_batch_size=None, batch_buckets=None,
        steps=steps, projection=projection,
        record_params=True, backend=backend,
    )
    result = solver.run(x0, jax.random.key(int(seed)), args)
    params = np.asarray(jax.device_get(result.history.params))
    if problem == 'quadratic':
        d = params - np.asarray(jax.device_get(args['target']))
        gaps = 0.5 * np.sum(np.asarray(jax.device_get(args['curvature'])) * d**2, axis=-1)
        exact = np.asarray(jax.device_get(args['target']))
    else:
        a = np.asarray(jax.device_get(args['target']))
        exact = np.array([0.60, 0.35, 0.05, 0.0], dtype=params.dtype)
        gaps = 0.5 * np.sum((params-a)**2, axis=-1) - 0.5 * np.sum((exact-a)**2)
        if np.any(params < -1e-7) or not np.allclose(params.sum(axis=1), 1.0, atol=1e-7):
            raise AssertionError('Simplex feasibility failure')
    if not np.all(np.isfinite(gaps)):
        raise FloatingPointError(f'Nonfinite gap: {problem}, lr={lr}, batch={batch}, seed={seed}')
    counts = np.asarray(jax.device_get(result.history.stats.batch_size), dtype=int)
    if not np.all(counts == batch):
        raise AssertionError('Batch size did not match requested fixed minibatch')
    return gaps, counts.cumsum(), params[-1], float(np.linalg.norm(params[-1]-exact))


def write_csv(path, fields, rows):
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot_problem(out, problem, lrs, batches, steps, curves, axis):
    fig, axs = plt.subplots(2, 2, figsize=(11, 7), sharey=True)
    for ax, b in zip(axs.ravel(), batches):
        for lr in lrs:
            v = curves[(problem, lr, b)]
            x = np.arange(1, steps+1) if axis == 'iteration' else v['samples']
            ax.plot(x, v['mean'], label=f'lr={lr:g}', lw=1.7)
            # Sparse SD errorbars keep the figure readable.
            idx = np.unique(np.linspace(0, steps-1, min(10, steps), dtype=int))
            ax.errorbar(x[idx], v['mean'][idx], yerr=v['sd'][idx], fmt='none', capsize=2, alpha=.45)
        ax.set_title(f'Batch size = {b}')
        ax.set_xlabel('Iterations' if axis == 'iteration' else 'Samples per run')
        ax.set_ylabel('Expected objective gap')
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.suptitle(f'{problem.replace("_", " ").title()}: mean ± SD across 10 seeds')
    fig.tight_layout()
    path = out / f'{problem}_{axis}.png'
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', default='results_2_1')
    p.add_argument('--steps', type=int, default=400)
    p.add_argument('--lrs', type=float, nargs='+', default=[0.01, 0.03, 0.1, 0.3])
    p.add_argument('--batches', type=int, nargs='+', default=[8, 32, 64, 128])
    p.add_argument('--seeds', type=int, nargs='+', default=list(range(10)))
    p.add_argument('--backend', choices=['cpu','gpu'], default='gpu')
    p.add_argument('--dtype', choices=['float32','float64'], default='float64')
    a = p.parse_args()
    if a.steps < 1 or min(a.batches) < 1 or not a.seeds:
        p.error('steps, batch sizes and seed count must be positive')
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    jdtype = jnp.float64 if a.dtype == 'float64' else jnp.float32
    problems = ['quadratic', 'simplex']
    raw_rows, summary_rows, curves = [], [], {}
    total = len(problems)*len(a.lrs)*len(a.batches)*len(a.seeds)
    done = 0
    for problem in problems:
        for lr in a.lrs:
            for batch in a.batches:
                all_gaps, all_samples = [], []
                for seed in a.seeds:
                    gaps, samples, solution, l2 = run_one(problem, lr, batch, seed, a.steps, jdtype, a.backend)
                    all_gaps.append(gaps)
                    all_samples.append(samples)
                    for t, (gap, sample_count) in enumerate(zip(gaps, samples), start=1):
                        raw_rows.append(dict(problem=problem, learning_rate=lr, batch_size=batch, seed=seed,
                                             iteration=t, samples=int(sample_count), expected_objective_gap=float(gap)))
                    done += 1
                    print(f'[{done}/{total}] {problem} lr={lr:g} batch={batch} seed={seed} gap={gaps[-1]:.6g}', flush=True)
                stack = np.stack(all_gaps)
                mean = stack.mean(axis=0)
                sd = stack.std(axis=0, ddof=1) if len(a.seeds)>1 else np.zeros_like(mean)
                samples = all_samples[0]
                curves[(problem,lr,batch)] = dict(mean=mean,sd=sd,samples=samples)
                summary_rows.append(dict(problem=problem, learning_rate=lr, batch_size=batch,
                    seeds=len(a.seeds), steps=a.steps, samples_per_run=int(samples[-1]),
                    final_gap_mean=float(mean[-1]), final_gap_sd=float(sd[-1])))
                # Checkpoint after every completed configuration.
                write_csv(out/'summary.csv', list(summary_rows[0]), summary_rows)
    write_csv(out/'per_seed_history.csv', list(raw_rows[0]), raw_rows)
    for problem in problems:
        plot_problem(out, problem, a.lrs, a.batches, a.steps, curves, 'iteration')
        plot_problem(out, problem, a.lrs, a.batches, a.steps, curves, 'samples')
    best = {}
    for problem in problems:
        subset = [r for r in summary_rows if r['problem']==problem]
        best[problem] = min(subset, key=lambda r: r['final_gap_mean'])
    with open(out/'best_configs.json','w') as f:
        json.dump(dict(settings=vars(a),best_by_final_mean_gap=best),f,indent=2)
    print('\nBest configuration by mean final objective gap:')
    print(json.dumps(best, indent=2))
    print('Output:', out.resolve())

if __name__ == '__main__':
    main()
