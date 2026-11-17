from functools import partial

import jax
import jax.numpy as jnp
from flax import struct


@struct.dataclass
class State:
    key: jax.Array
    x_t: jax.Array
    y_t: jax.Array


def index_from_ladder(y: jax.Array, h_ladder):
    return jnp.minimum(
        (h_ladder[:-1, None] <= -y[None]) & (h_ladder[1:, None] > -y[None]),
        axis=0,
    )


@partial(jax.jit, static_argnames=("kernel_eval", "kernel_samp", "target"))
def step(
    carry: State,
    _,
    k: int,  # Current chain to update
    h_ladder: jax.Array,
    t_ladder: jax.Array,
    kernel_eval: callable,
    kernel_samp: callable,
    log_target: callable,
    p_ee: jax.Array,  # Probability of an equi-energy jump
    buffer: jax.Array,
    buffer_mask: jax.Array,  # buffer_mask[i][j][k] = True if buffer is filled at position i, j, k
):
    # This assumes `buffer` has already been filled
    x_t = carry.x_t
    y_t = carry.y_t
    key = carry.key

    # Compute the energy function for the i-th chain
    h_t = -jnp.maximum(y_t, h_ladder) / t_ladder
    buffer_index = index_from_ladder(y_t[k], h_ladder)
    k_n, k_b, k_k, k_c, k_u = jax.random.split(key, 5)

    num_chains = len(x_t)

    def next_kernel(key, x_t_k):
        x_tp1_k, logp_t_tp1_k = kernel_samp(key, x_t_k)
        logp_tp1_t_k = kernel_eval(x_tp1_k, x_t_k)

        # Compute the target and the acceptance ratio
        y_tp1_k = log_target(x_tp1_k)
        h_tp1 = -jnp.maximum(y_tp1_k, h_ladder[k]) / t_ladder[k]
        log_alpha = (-h_tp1) + logp_tp1_t_k - logp_t_tp1_k - (-h_t[k])

        return x_tp1_k, y_tp1_k, log_alpha

    def next_buffer(key, _):
        # Sample from the buffer
        sample_index = jax.random.categorical(
            key, logits=jnp.where(buffer_mask[k], 1.0, -jnp.inf)
        )

        # Sample from the higher-level buffer (uniformly) and compute the target
        x_tp1_k = buffer[k + 1, buffer_index, sample_index]
        y_tp1_k = log_target(x_tp1_k)
        h_tp1 = -jnp.maximum(y_tp1_k, h_ladder[k]) / t_ladder[k]

        # Compute the acceptance rate: for non-swap moves, it is the standard MH
        # For swap moves, it is the ratio between the product distributions

        log_alpha = (
            -h_tp1
            - jnp.maximum(y_t, h_ladder[k + 1]) / t_ladder[k + 1]
            + h_t[k]
            + jnp.maximum(y_tp1_k, h_ladder[k]) / t_ladder[k]
        )

        return x_tp1_k, y_tp1_k, log_alpha

    # Sample a Bernoulli variable for each chain
    should_ee_jump = (
        jax.random.bernoulli(k_b, p=p_ee, dtype=bool)
        & (k < num_chains)
        & buffer_mask[k + 1, buffer_index].any(axis=1)
    )

    x_tp1_k, y_tp1_k, log_alpha = jax.lax.cond(
        should_ee_jump,
        lambda: next_buffer(k_k),
        lambda: next_kernel(k_c, x_t[k]),
    )
    # Compute the acceptance probabilities
    u = jax.random.uniform(k_u)
    should_move = u <= log_alpha
    x_tp1 = x_t.at[k].set(jnp.where(should_move[None], x_tp1_k, x_t[k]))
    y_tp1 = y_t.at[k].set(jnp.where(should_move, y_tp1_k, y_t[k]))

    return State(key=k_n, x_t=x_tp1, y_t=y_tp1), should_move


def sample(
    key: jax.Array,
    x_o: jax.Array,
    t_ladder: jax.Array,
    h_ladder: jax.Array,
    kernel_eval: callable,
    kernel_samp: callable,
    log_target: callable,
    burn_in: int,
    pre_fill: int,
    chain_size: int,
    p_ee: float = 0.1,
):
    num_chains, d = x_o.shape

    # Create the buffer based on the number of chains
    buffer = jnp.empty((num_chains, num_chains, chain_size, d), jnp.float32)
    buffer_mask = jnp.zeros((num_chains, num_chains, chain_size), jnp.bool)

    # Set the initial state in the buffer
    y_o = jax.vmap(log_target, in_axes=(0,), out_axes=0)(x_o)
    index = index_from_ladder(y_o, h_ladder)
    buffer = buffer.at[:, index, 0, :].set(x_o)
    buffer_mask = buffer_mask.at[:, index, 0].set(True)

    step_p = partial(
        step,
        h_ladder=h_ladder,
        t_ladder=t_ladder,
        kernel_eval=kernel_eval,
        kernel_samp=kernel_samp,
        log_target=log_target,
        p_ee=p_ee,
    )

    # Sample from the chain
    @jax.jit
    def next(carry: tuple[int, State, jax.Array, jax.Array], _):
        idx, state, buffer, buffer_mask = carry

        def next_level(
            i: jax.Array, carry: tuple[State, jax.Array, jax.Array]
        ):
            state, buffer, buffer_mask = carry

            k = num_chains - i

            should_advance = idx > (num_chains - k) * (pre_fill + burn_in)

            (state, has_moved), _ = jax.lax.cond(
                should_advance,
                lambda: step_p(
                    state, k, buffer=buffer, buffer_mask=buffer_mask
                ),
                lambda: (state, jnp.array(False)),
            )

            should_update_buffer = (
                idx > (num_chains - k) * (pre_fill + burn_in) + burn_in
            )

            def update_buffer(buffer, buffer_mask):
                # In continuous spaces, the likelihood of sampling
                # two identical states is null; we hence avoid verifying
                # whether the current state was already sampled

                # Check the latest filled index
                latest_available_entry = jnp.argmin(buffer_mask)
                index = index_from_ladder(state.y_t, h_ladder)

                # Update the buffer and the buffer's mask
                buffer = buffer.at[k, index, latest_available_entry].set(
                    state.x_t
                )
                buffer_mask = buffer_mask.at[
                    k, index, latest_available_entry
                ].set(True)

                return buffer, buffer_mask

            buffer, buffer_mask = jax.lax.cond(
                should_update_buffer & has_moved,
                lambda: update_buffer(buffer, buffer_mask),
                lambda: (buffer, buffer_mask),
            )
            return (state, buffer, buffer_mask), _

        (state, buffer, buffer_mask), _ = jax.lax.fori_loop(
            lower=0,
            upper=num_chains,
            body_fun=next_level,
            init_val=(state, buffer, buffer_mask),
        )

        return (idx + 1, state, buffer, buffer_mask), state.x_t

    # Sample from the chain after filling up the buffer
    _, samples = jax.lax.scan(
        next,
        init=(
            0,
            State(x_t=x_o, y_t=y_o, key=key),
            buffer,
            buffer_mask,
        ),
        length=chain_size,
    )
    return samples
