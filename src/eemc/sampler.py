from collections.abc import Callable
from functools import partial

import jax
import jax.numpy as jnp
from flax import struct


@struct.dataclass
class State:
    key: jax.Array
    x_t: jax.Array
    y_t: jax.Array


def index_from_ladder(y: jax.Array, h_bins: jax.Array) -> jax.Array:
    num_bins = h_bins.shape[0] - 1
    idx = jnp.searchsorted(h_bins, -y, side="right") - 1
    return jnp.clip(idx, 0, num_bins - 1)


def _log_tempered_target(
    y: jax.Array, h_threshold: jax.Array, temperature: jax.Array
) -> jax.Array:
    return -jnp.maximum(-y, h_threshold) / temperature


@partial(jax.jit, static_argnames=("kernel_eval", "kernel_samp", "log_target"))
def step(
    carry: State,
    k: jax.Array,
    h_bins: jax.Array,
    t_ladder: jax.Array,
    kernel_eval: Callable,
    kernel_samp: Callable,
    log_target: Callable,
    p_ee: float,
    buffer: jax.Array,
    ring_counts: jax.Array,
) -> tuple[State, jax.Array]:
    x_t = carry.x_t
    y_t = carry.y_t
    key = carry.key

    num_chains = x_t.shape[0]
    h_thresholds = h_bins[:num_chains]
    k_next = jnp.minimum(k + 1, num_chains - 1)

    x_k = x_t[k]
    y_k = y_t[k]
    bin_idx = index_from_ladder(y_k, h_bins)

    k_n, k_b, k_k, k_c, k_u = jax.random.split(key, 5)

    def next_kernel(subkey: jax.Array):
        x_prop, logp_fwd = kernel_samp(subkey, x_k, k)
        logp_bwd = kernel_eval(x_prop, x_k, k)

        y_prop = log_target(x_prop)
        log_pi_prop = _log_tempered_target(y_prop, h_thresholds[k], t_ladder[k])
        log_pi_curr = _log_tempered_target(y_k, h_thresholds[k], t_ladder[k])

        log_alpha = log_pi_prop - log_pi_curr + logp_bwd - logp_fwd
        return x_prop, y_prop, log_alpha

    def next_buffer(subkey: jax.Array):
        buffer_cap = buffer.shape[2]
        n_valid = jnp.minimum(ring_counts[k_next, bin_idx], buffer_cap)
        sample_idx = jax.random.randint(
            subkey, (), 0, jnp.maximum(n_valid, 1)
        )

        x_prop = buffer[k_next, bin_idx, sample_idx]
        y_prop = log_target(x_prop)

        log_pi_k_prop = _log_tempered_target(y_prop, h_thresholds[k], t_ladder[k])
        log_pi_k_curr = _log_tempered_target(y_k, h_thresholds[k], t_ladder[k])
        log_pi_next_curr = _log_tempered_target(
            y_k, h_thresholds[k_next], t_ladder[k_next]
        )
        log_pi_next_prop = _log_tempered_target(
            y_prop, h_thresholds[k_next], t_ladder[k_next]
        )

        log_alpha = (
            log_pi_k_prop
            + log_pi_next_curr
            - log_pi_k_curr
            - log_pi_next_prop
        )
        return x_prop, y_prop, log_alpha

    should_ee_jump = (
        jax.random.bernoulli(k_b, p=p_ee)
        & (k < num_chains - 1)
        & (ring_counts[k_next, bin_idx] > 0)
    )

    x_prop, y_prop, log_alpha = jax.lax.cond(
        should_ee_jump,
        lambda: next_buffer(k_k),
        lambda: next_kernel(k_c),
    )

    u = jax.random.uniform(k_u)
    should_move = jnp.log(u) <= log_alpha
    x_tp1 = x_t.at[k].set(jnp.where(should_move, x_prop, x_k))
    y_tp1 = y_t.at[k].set(jnp.where(should_move, y_prop, y_k))

    return State(key=k_n, x_t=x_tp1, y_t=y_tp1), should_move


def sample(
    key: jax.Array,
    x_o: jax.Array,
    t_ladder: jax.Array,
    h_ladder: jax.Array,
    kernel_eval: Callable,
    kernel_samp: Callable,
    log_target: Callable,
    burn_in: int,
    pre_fill: int,
    chain_size: int,
    p_ee: float = 0.1,
    max_ring_size: int | None = None,
) -> jax.Array:
    num_chains = x_o.shape[0]
    state_shape = x_o.shape[1:]

    h_bins = jnp.concatenate(
        [h_ladder, jnp.array([jnp.inf], dtype=h_ladder.dtype)]
    )

    total_steps = (num_chains - 1) * (burn_in + pre_fill) + burn_in + chain_size
    buffer_cap = total_steps if max_ring_size is None else min(total_steps, max_ring_size)

    buffer = jnp.empty((num_chains, num_chains, buffer_cap, *state_shape), dtype=x_o.dtype)
    ring_counts = jnp.zeros((num_chains, num_chains), dtype=jnp.int32)

    y_o = jax.vmap(log_target)(x_o)

    step_p = partial(
        step,
        h_bins=h_bins,
        t_ladder=t_ladder,
        kernel_eval=kernel_eval,
        kernel_samp=kernel_samp,
        log_target=log_target,
        p_ee=p_ee,
    )

    @jax.jit
    def scan_step(carry: tuple[jax.Array, State, jax.Array, jax.Array], _):
        n, state, buffer, ring_counts = carry

        def next_level(i: jax.Array, inner_carry: tuple[State, jax.Array, jax.Array]):
            state, buffer, ring_counts = inner_carry
            k = (num_chains - 1) - i
            start_offset = (num_chains - 1 - k) * (burn_in + pre_fill)

            should_advance = n > start_offset
            state, _ = jax.lax.cond(
                should_advance,
                lambda: step_p(state, k, buffer=buffer, ring_counts=ring_counts),
                lambda: (state, jnp.array(False)),
            )

            should_update_buffer = n > start_offset + burn_in

            def update_buffer(buf, counts):
                bin_idx = index_from_ladder(state.y_t[k], h_bins)
                pos = counts[k, bin_idx] % buffer_cap
                buf = buf.at[k, bin_idx, pos].set(state.x_t[k])
                counts = counts.at[k, bin_idx].add(1)
                return buf, counts

            buffer, ring_counts = jax.lax.cond(
                should_update_buffer,
                lambda: update_buffer(buffer, ring_counts),
                lambda: (buffer, ring_counts),
            )
            return state, buffer, ring_counts

        state, buffer, ring_counts = jax.lax.fori_loop(
            lower=0,
            upper=num_chains,
            body_fun=next_level,
            init_val=(state, buffer, ring_counts),
        )

        return (n + 1, state, buffer, ring_counts), state.x_t

    _, samples = jax.lax.scan(
        scan_step,
        init=(
            jnp.array(1, dtype=jnp.int32),
            State(x_t=x_o, y_t=y_o, key=key),
            buffer,
            ring_counts,
        ),
        length=total_steps,
    )
    return samples[-chain_size:]


def sample_pt(
    key: jax.Array,
    x_o: jax.Array,
    t_ladder: jax.Array,
    kernel_eval: Callable,
    kernel_samp: Callable,
    log_target: Callable,
    burn_in: int,
    chain_size: int,
    h_ladder: jax.Array | None = None,
    p_swap: float = 1.0,
) -> jax.Array:
    num_chains = x_o.shape[0]
    h_thresholds = (
        jnp.full((num_chains,), -jnp.inf, dtype=t_ladder.dtype)
        if h_ladder is None
        else h_ladder
    )
    y_o = jax.vmap(log_target)(x_o)
    total_steps = burn_in + chain_size

    @jax.jit
    def scan_step(carry: tuple[jax.Array, State], _):
        step_idx, state = carry
        key, k_mh, k_swap_gate, k_swap = jax.random.split(state.key, 4)

        def mh_chain(k_sub, x_k, y_k, chain_idx):
            k_prop, k_u = jax.random.split(k_sub)
            x_prop, logp_fwd = kernel_samp(k_prop, x_k, chain_idx)
            logp_bwd = kernel_eval(x_prop, x_k, chain_idx)
            y_prop = log_target(x_prop)
            log_pi_prop = _log_tempered_target(
                y_prop, h_thresholds[chain_idx], t_ladder[chain_idx]
            )
            log_pi_curr = _log_tempered_target(
                y_k, h_thresholds[chain_idx], t_ladder[chain_idx]
            )
            log_alpha = log_pi_prop - log_pi_curr + logp_bwd - logp_fwd
            move = jnp.log(jax.random.uniform(k_u)) <= log_alpha
            return jnp.where(move, x_prop, x_k), jnp.where(move, y_prop, y_k)

        mh_keys = jax.random.split(k_mh, num_chains)
        x_t, y_t = jax.vmap(mh_chain)(
            mh_keys, state.x_t, state.y_t, jnp.arange(num_chains)
        )

        def swap_step(x_arr, y_arr):
            parity = step_idx % 2

            def pair_body(idx, xy):
                x_cur, y_cur = xy
                i = 2 * idx + parity
                do_pair = i < num_chains - 1

                def try_swap():
                    j = i + 1
                    log_alpha = (
                        _log_tempered_target(y_cur[j], h_thresholds[i], t_ladder[i])
                        + _log_tempered_target(y_cur[i], h_thresholds[j], t_ladder[j])
                        - _log_tempered_target(y_cur[i], h_thresholds[i], t_ladder[i])
                        - _log_tempered_target(y_cur[j], h_thresholds[j], t_ladder[j])
                    )
                    accept = jnp.log(jax.random.uniform(jax.random.fold_in(k_swap, i))) <= log_alpha
                    x_swapped = x_cur.at[i].set(x_cur[j]).at[j].set(x_cur[i])
                    y_swapped = y_cur.at[i].set(y_cur[j]).at[j].set(y_cur[i])
                    return (
                        jnp.where(accept, x_swapped, x_cur),
                        jnp.where(accept, y_swapped, y_cur),
                    )

                return jax.lax.cond(do_pair, try_swap, lambda: (x_cur, y_cur))

            return jax.lax.fori_loop(0, (num_chains + 1) // 2, pair_body, (x_arr, y_arr))

        do_swap = jax.random.bernoulli(k_swap_gate, p=p_swap)
        x_t, y_t = jax.lax.cond(
            do_swap, lambda: swap_step(x_t, y_t), lambda: (x_t, y_t)
        )
        return (step_idx + 1, State(key=key, x_t=x_t, y_t=y_t)), x_t

    _, samples = jax.lax.scan(
        scan_step,
        init=(jnp.array(0, dtype=jnp.int32), State(key=key, x_t=x_o, y_t=y_o)),
        length=total_steps,
    )
    return samples[-chain_size:]
