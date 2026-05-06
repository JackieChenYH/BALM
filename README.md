# BALM
A Bayesian Adaptive Latent Mixture Model for Zero-Inflated Weighted Brain Connectome Analysis


**Abstract**:

Replicated weighted networks often contain many structural zeros together with heterogeneous non-zero edge strengths. This occurs in structural connectomics, where subjects may express overlapping rather than discrete connectivity patterns. We propose a Bayesian adaptive latent mixture model for zero-inflated weighted networks. The model represents each subject through a simplex mixture of shared low-rank latent score matrices and combines this representation with a hurdle likelihood that separates edge existence from conditional edge strength. A sparsity-coupling parameter allows absent edges either to be independent of, or informative about, latent connectivity. Posterior computation uses transformed Hamiltonian Monte Carlo on unconstrained coordinates, and the number of templates is selected using predictive fit, held-out link prediction and template stability. For an identifiable quotient-space estimand, we establish posterior consistency, local asymptotic normality, a Bernstein--von Mises approximation and predictive consistency under a fixed-template regime. Simulations show gains over topology-only baselines when subject memberships are mixed or sparsity is structure informed. In Human Connectome Project structural connectomes, the method recovers stable latent score patterns and heterogeneous subject-level mixtures. Behavioural analyses are used as exploratory annotations of the recovered templates rather than as confirmatory biomarker claims.

Author:
Hsin-Hsiung Huan, Yuh-Haur Chen, Teng Zhang
