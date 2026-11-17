import unittest

import jax
import jax.numpy as jnp

from eemc.sampler import index_from_ladder, sample, sample_pt


class TestEEMCSampler(unittest.TestCase):
    def test_index_from_ladder_boundaries(self):
        h_bins = jnp.array([0.0, 2.0, 5.0, jnp.inf])
        self.assertEqual(int(index_from_ladder(jnp.array(1.0), h_bins)), 0)
        self.assertEqual(int(index_from_ladder(jnp.array(-0.0), h_bins)), 0)
        self.assertEqual(int(index_from_ladder(jnp.array(-1.999), h_bins)), 0)
        self.assertEqual(int(index_from_ladder(jnp.array(-2.0), h_bins)), 1)
        self.assertEqual(int(index_from_ladder(jnp.array(-4.999), h_bins)), 1)
        self.assertEqual(int(index_from_ladder(jnp.array(-5.0), h_bins)), 2)
        self.assertEqual(int(index_from_ladder(jnp.array(-1e6), h_bins)), 2)

    def test_unimodal_gaussian(self):
        def log_target(x):
            return -0.5 * jnp.sum((x - 1.0) ** 2)

        def kernel_samp(key, x, _k):
            z = jax.random.normal(key, shape=x.shape)
            return x + z, -0.5 * jnp.sum(z**2)

        def kernel_eval(x_from, x_to, _k):
            return -0.5 * jnp.sum((x_to - x_from) ** 2)

        samples = sample(
            key=jax.random.PRNGKey(1),
            x_o=jnp.zeros((2, 1)),
            t_ladder=jnp.array([1.0, 3.0]),
            h_ladder=jnp.array([0.0, 1.5]),
            kernel_eval=kernel_eval,
            kernel_samp=kernel_samp,
            log_target=log_target,
            burn_in=100,
            pre_fill=100,
            chain_size=1500,
            p_ee=0.2,
            max_ring_size=500,
        )
        self.assertEqual(samples.shape, (1500, 2, 1))
        x0 = samples[:, 0, 0]
        self.assertAlmostEqual(float(jnp.mean(x0)), 1.0, delta=0.25)
        self.assertAlmostEqual(float(jnp.std(x0)), 1.0, delta=0.25)

    def test_bimodal_1d_gaussian_mixture(self):
        sigma_mode = 0.4
        var = sigma_mode**2

        def log_target(x):
            lp1 = -0.5 * jnp.sum((x + 4.0) ** 2) / var
            lp2 = -0.5 * jnp.sum((x - 4.0) ** 2) / var
            return jax.scipy.special.logsumexp(jnp.array([lp1, lp2])) - jnp.log(2.0)

        t_ladder = jnp.array([1.0, 4.0, 16.0, 64.0])
        h_ladder = jnp.array([0.0, 2.0, 6.0, 15.0])
        sigmas = sigma_mode * jnp.sqrt(t_ladder) * 1.5

        def kernel_samp(key, x, k):
            z = jax.random.normal(key, shape=x.shape)
            s = sigmas[k]
            return x + s * z, -0.5 * jnp.sum(z**2) - x.shape[0] * jnp.log(s)

        def kernel_eval(x_from, x_to, k):
            s = sigmas[k]
            return -0.5 * jnp.sum(((x_to - x_from) / s) ** 2) - x_from.shape[0] * jnp.log(s)

        x_o = jnp.full((4, 1), -4.0)
        samples = sample(
            key=jax.random.PRNGKey(7),
            x_o=x_o,
            t_ladder=t_ladder,
            h_ladder=h_ladder,
            kernel_eval=kernel_eval,
            kernel_samp=kernel_samp,
            log_target=log_target,
            burn_in=300,
            pre_fill=400,
            chain_size=4000,
            p_ee=0.2,
        )
        x0 = samples[:, 0, 0]
        frac_right = float(jnp.mean(x0 > 0.0))
        self.assertGreater(frac_right, 0.35)
        self.assertLess(frac_right, 0.65)
        self.assertAlmostEqual(float(jnp.std(x0)), 4.02, delta=0.35)

    def test_pt_and_truncated_pt_bimodal(self):
        sigma_mode = 0.4
        var = sigma_mode**2

        def log_target(x):
            lp1 = -0.5 * jnp.sum((x + 4.0) ** 2) / var
            lp2 = -0.5 * jnp.sum((x - 4.0) ** 2) / var
            return jax.scipy.special.logsumexp(jnp.array([lp1, lp2])) - jnp.log(2.0)

        t_ladder = jnp.array([1.0, 4.0, 16.0, 64.0])
        h_ladder = jnp.array([0.0, 2.0, 6.0, 15.0])
        sigmas = sigma_mode * jnp.sqrt(t_ladder) * 1.5

        def kernel_samp(key, x, k):
            z = jax.random.normal(key, shape=x.shape)
            s = sigmas[k]
            return x + s * z, -0.5 * jnp.sum(z**2) - x.shape[0] * jnp.log(s)

        def kernel_eval(x_from, x_to, k):
            s = sigmas[k]
            return -0.5 * jnp.sum(((x_to - x_from) / s) ** 2) - x_from.shape[0] * jnp.log(s)

        x_o = jnp.full((4, 1), -4.0)
        for h_lad in (None, h_ladder):
            samples = sample_pt(
                key=jax.random.PRNGKey(11),
                x_o=x_o,
                t_ladder=t_ladder,
                kernel_eval=kernel_eval,
                kernel_samp=kernel_samp,
                log_target=log_target,
                burn_in=1000,
                chain_size=4000,
                h_ladder=h_lad,
                p_swap=0.5,
            )
            x0 = samples[:, 0, 0]
            frac_right = float(jnp.mean(x0 > 0.0))
            self.assertGreater(frac_right, 0.30)
            self.assertLess(frac_right, 0.70)
            self.assertAlmostEqual(float(jnp.std(x0)), 4.02, delta=0.45)


if __name__ == "__main__":
    unittest.main()
