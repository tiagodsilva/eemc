import base64
import io
import os
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from eemc.sampler import sample, sample_pt


def make_sparse_gmm(grid_coords=(-6.0, -2.0, 2.0, 6.0), sigma=0.15):
    xs = jnp.array(grid_coords, dtype=jnp.float32)
    gx, gy = jnp.meshgrid(xs, xs, indexing="ij")
    means = jnp.stack([gx.ravel(), gy.ravel()], axis=-1)  # (16, 2)
    num_modes = means.shape[0]
    var = sigma**2

    def log_target(x: jax.Array) -> jax.Array:
        diff = x[None, :] - means  # (M, 2)
        log_components = -0.5 * jnp.sum(diff**2, axis=-1) / var - jnp.log(
            2.0 * jnp.pi * var
        )
        return jax.scipy.special.logsumexp(log_components) - jnp.log(num_modes)

    return means, sigma, log_target


def run_mh_baseline(key, x_init, log_target, proposal_sigma, num_steps):
    def mh_step(carry, _):
        subkey, x, logp = carry
        subkey, k_prop, k_u = jax.random.split(subkey, 3)
        x_prop = x + proposal_sigma * jax.random.normal(k_prop, shape=x.shape)
        logp_prop = log_target(x_prop)
        accept = jnp.log(jax.random.uniform(k_u)) <= (logp_prop - logp)
        x_next = jnp.where(accept, x_prop, x)
        logp_next = jnp.where(accept, logp_prop, logp)
        return (subkey, x_next, logp_next), x_next

    _, chain = jax.lax.scan(
        mh_step,
        init=(key, x_init, log_target(x_init)),
        length=num_steps,
    )
    return chain


def evaluate_samples(samples: jax.Array, means: jax.Array, sigma: float):
    dists = jnp.linalg.norm(samples[:, None, :] - means[None, :, :], axis=-1)
    nearest_mode = jnp.argmin(dists, axis=-1)
    min_dist = jnp.min(dists, axis=-1)
    in_mode = min_dist <= 5.0 * sigma

    num_modes = means.shape[0]
    counts = jnp.bincount(
        jnp.where(in_mode, nearest_mode, num_modes), length=num_modes + 1
    )[:num_modes]
    total_assigned = jnp.maximum(jnp.sum(counts), 1)
    est_weights = counts / total_assigned
    true_weight = 1.0 / num_modes

    modes_visited = int(jnp.sum(counts > 0))
    tv_error = 0.5 * float(jnp.sum(jnp.abs(est_weights - true_weight)))
    mean_est = jnp.mean(samples, axis=0)
    std_est = jnp.std(samples, axis=0)
    return modes_visited, tv_error, mean_est, std_est, counts, est_weights


def supports_kitty_graphics() -> bool:
    if not sys.stdout.isatty():
        return False
    term = os.environ.get("TERM", "").lower()
    term_program = os.environ.get("TERM_PROGRAM", "").lower()
    return (
        bool(os.environ.get("KITTY_WINDOW_ID"))
        or bool(os.environ.get("GHOSTTY_RESOURCES_DIR"))
        or "kitty" in term
        or term_program in {"ghostty", "kitty", "wezterm"}
    )


def display_png_in_terminal(png_bytes: bytes) -> None:
    encoded = base64.standard_b64encode(png_bytes).decode("ascii")
    chunk_size = 4096
    chunks = [
        encoded[i : i + chunk_size] for i in range(0, len(encoded), chunk_size)
    ]
    for idx, chunk in enumerate(chunks):
        more = 1 if idx < len(chunks) - 1 else 0
        if idx == 0:
            sys.stdout.write(f"\x1b_Ga=T,f=100,m={more};{chunk}\x1b\\")
        else:
            sys.stdout.write(f"\x1b_Gm={more};{chunk}\x1b\\")
    sys.stdout.write("\n")
    sys.stdout.flush()


def plot_results(
    mh_samples: np.ndarray,
    mh_weights: np.ndarray,
    methods: list[tuple[str, str, np.ndarray, np.ndarray]],
    means: np.ndarray,
    output_path: Path,
) -> bytes:
    n_rows = len(methods)
    fig, axes = plt.subplots(n_rows, 3, figsize=(8, 2.8 * n_rows), dpi=150)

    num_modes = len(means)
    idx = np.arange(num_modes)
    width = 0.38

    for row, (label, color, samps, weights) in enumerate(methods):
        # Panel 1: Standard MH scatter
        ax = axes[row, 0]
        ax.scatter(
            mh_samples[:, 0],
            mh_samples[:, 1],
            s=4,
            alpha=0.25,
            color="tab:red",
            label="MH samples",
        )
        ax.scatter(
            means[:, 0],
            means[:, 1],
            marker="x",
            s=40,
            color="black",
            linewidths=1.2,
            label="True modes",
        )
        ax.set_title("Standard MH (T = 1)", fontsize=9)
        ax.set_xlabel("$x_1$", fontsize=8)
        ax.set_ylabel("$x_2$", fontsize=8)
        ax.set_xlim(-8, 8)
        ax.set_ylim(-8, 8)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        if row == 0:
            ax.legend(loc="upper right", fontsize=7)

        # Panel 2: Method scatter
        ax = axes[row, 1]
        ax.scatter(
            samps[:, 0],
            samps[:, 1],
            s=4,
            alpha=0.25,
            color=color,
            label=f"{label} samples",
        )
        ax.scatter(
            means[:, 0],
            means[:, 1],
            marker="x",
            s=40,
            color="black",
            linewidths=1.2,
            label="True modes",
        )
        ax.set_title(f"{label} (6 chains, $T_0=1$)", fontsize=9)
        ax.set_xlabel("$x_1$", fontsize=8)
        ax.set_ylabel("$x_2$", fontsize=8)
        ax.set_xlim(-8, 8)
        ax.set_ylim(-8, 8)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=7)

        # Panel 3: Mode weights comparison
        ax = axes[row, 2]
        ax.bar(
            idx - width / 2,
            mh_weights,
            width,
            color="tab:red",
            alpha=0.8,
            label="MH",
        )
        ax.bar(
            idx + width / 2,
            weights,
            width,
            color=color,
            alpha=0.8,
            label=label,
        )
        ax.axhline(
            1.0 / num_modes,
            color="black",
            linestyle="--",
            linewidth=1.0,
            label=f"True ({1.0 / num_modes:.4f})",
        )
        ax.set_title(f"Mode Weights ({label} vs MH)", fontsize=9)
        ax.set_xlabel("Mode index", fontsize=8)
        ax.set_ylabel("Empirical weight", fontsize=8)
        ax.set_xticks(idx[::2])
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(loc="upper right", fontsize=7)

    fig.suptitle(
        "Sparse 2D Gaussian Mixture (16 modes, $\\sigma = 0.15$)",
        fontsize=11,
        fontweight="bold",
    )
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    png_bytes = buf.getvalue()
    output_path.write_bytes(png_bytes)
    return png_bytes


def main():
    means, sigma, log_target = make_sparse_gmm()
    num_modes = means.shape[0]
    true_std = float(jnp.sqrt(jnp.mean(means[:, 0] ** 2) + sigma**2))

    h_min = float(-log_target(means[0]))
    t_ladder = jnp.array([1.0, 2.5, 6.0, 15.0, 40.0, 100.0])
    h_ladder = jnp.array([h_min - 0.1, 2.0, 5.0, 12.0, 25.0, 50.0])
    num_chains = len(t_ladder)

    proposal_sigmas = sigma * jnp.sqrt(t_ladder) * 1.5

    def kernel_samp(key, x, k):
        s = proposal_sigmas[k]
        z = jax.random.normal(key, shape=x.shape)
        prop = x + s * z
        log_q = -0.5 * jnp.sum(z**2) - x.shape[0] * jnp.log(s)
        return prop, log_q

    def kernel_eval(x_from, x_to, k):
        s = proposal_sigmas[k]
        diff = (x_to - x_from) / s
        return -0.5 * jnp.sum(diff**2) - x_from.shape[0] * jnp.log(s)

    key = jax.random.PRNGKey(42)
    k_init, k_eemc, k_pt, k_tpt, k_mh = jax.random.split(key, 5)

    x_o = jnp.tile(
        jnp.array([-6.0, -6.0]), (num_chains, 1)
    ) + 0.1 * jax.random.normal(k_init, (num_chains, 2))

    burn_in = 500
    pre_fill = 1000
    chain_size = 10000
    total_steps = (num_chains - 1) * (burn_in + pre_fill) + burn_in + chain_size
    pt_burn_in = total_steps - chain_size

    eemc_samples = sample(
        key=k_eemc,
        x_o=x_o,
        t_ladder=t_ladder,
        h_ladder=h_ladder,
        kernel_eval=kernel_eval,
        kernel_samp=kernel_samp,
        log_target=log_target,
        burn_in=burn_in,
        pre_fill=pre_fill,
        chain_size=chain_size,
        p_ee=0.2,
    )[:, 0, :]

    pt_samples = sample_pt(
        key=k_pt,
        x_o=x_o,
        t_ladder=t_ladder,
        kernel_eval=kernel_eval,
        kernel_samp=kernel_samp,
        log_target=log_target,
        burn_in=pt_burn_in,
        chain_size=chain_size,
        h_ladder=None,
        p_swap=0.2,
    )[:, 0, :]

    tpt_samples = sample_pt(
        key=k_tpt,
        x_o=x_o,
        t_ladder=t_ladder,
        kernel_eval=kernel_eval,
        kernel_samp=kernel_samp,
        log_target=log_target,
        burn_in=pt_burn_in,
        chain_size=chain_size,
        h_ladder=h_ladder,
        p_swap=0.2,
    )[:, 0, :]

    mh_samples = run_mh_baseline(
        k_mh,
        x_o[0],
        log_target,
        proposal_sigma=float(proposal_sigmas[0]),
        num_steps=total_steps,
    )[-chain_size:]

    print(f"Sparse 2D Gaussian Mixture ({num_modes} modes, sigma={sigma})")
    print(f"True mean: [0.00, 0.00], True marginal std: {true_std:.3f}\n")

    mh_res = evaluate_samples(mh_samples, means, sigma)
    eemc_res = evaluate_samples(eemc_samples, means, sigma)
    pt_res = evaluate_samples(pt_samples, means, sigma)
    tpt_res = evaluate_samples(tpt_samples, means, sigma)

    for name, (visited, tv_err, mean_est, std_est, counts, _) in [
        ("Standard MH (T=1)", mh_res),
        ("EEMC (6 chains)", eemc_res),
        ("PT (6 chains)", pt_res),
        ("Truncated PT (6 chains)", tpt_res),
    ]:
        print(f"--- {name} ---")
        print(f"  Modes visited : {visited}/{num_modes}")
        print(f"  Weight TV err : {tv_err:.4f} (0.0 = exact uniform)")
        print(
            f"  Sample mean   : [{float(mean_est[0]):.3f}, {float(mean_est[1]):.3f}]"
        )
        print(
            f"  Sample std    : [{float(std_est[0]):.3f}, {float(std_est[1]):.3f}]"
        )
        print(f"  Mode counts   : {counts.tolist()}\n")

    output_path = Path(__file__).with_name("sparse_gaussian_mixture.png")
    png_bytes = plot_results(
        np.asarray(mh_samples),
        np.asarray(mh_res[5]),
        [
            ("EEMC", "tab:blue", np.asarray(eemc_samples), np.asarray(eemc_res[5])),
            ("PT", "tab:green", np.asarray(pt_samples), np.asarray(pt_res[5])),
            (
                "Truncated PT",
                "tab:purple",
                np.asarray(tpt_samples),
                np.asarray(tpt_res[5]),
            ),
        ],
        np.asarray(means),
        output_path,
    )
    print(f"Saved plot to {output_path}")

    if supports_kitty_graphics():
        display_png_in_terminal(png_bytes)


if __name__ == "__main__":
    main()
