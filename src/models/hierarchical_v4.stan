// =============================================================================
// Hierarchical Model v4 — Species×Sex cross-classified groups (6 cells)
// =============================================================================
// What is hierarchical: alpha_g (intercept per species×sex cell, 6 groups)
// What is pooled:       beta_depth (single slope for bill depth)
//
// Rather than modelling species and sex as separate additive effects, this
// model treats every species×sex combination as a distinct group: Adelie-F,
// Adelie-M, Chinstrap-F, Chinstrap-M, Gentoo-F, Gentoo-M (6 total).
// Each cell gets its own intercept alpha_g drawn from a single shared
// Normal(mu_alpha, tau_alpha), so all 6 cells still borrow strength from one
// another through the hyperprior.
//
// The key difference from v3: there is no additive decomposition into a
// species effect plus a sex effect. The full species×sex interaction is
// captured directly in the group intercept, at the cost of one fewer
// structural assumption but also less interpretability (you cannot read off
// a "sex effect" or "species effect" separately).
//
// Cell index mapping (group = (species-1)*2 + sex + 1):
//   1 = Adelie-F,    2 = Adelie-M
//   3 = Chinstrap-F, 4 = Chinstrap-M
//   5 = Gentoo-F,    6 = Gentoo-M
//
// bill_length_i ~ Normal(alpha_g[i] + beta_depth*x_i, sigma)
// alpha_g       ~ Normal(mu_alpha, tau_alpha)    [hierarchical, 6 groups]
// mu_alpha      ~ Normal(0, 1)
// tau_alpha     ~ half-Cauchy(0, 1)
// beta_depth    ~ Normal(0, 1)                   [pooled]
// sigma         ~ Exponential(1)
// =============================================================================
data {
  int<lower=0> N;
  int<lower=0> G;              // number of groups = J * 2 = 6
  vector[N] bill_length;
  vector[N] bill_depth;
  array[N] int group;          // group index 1..G (species x sex cell)
}

parameters {
  real mu_alpha;
  real<lower=0> tau_alpha;

  vector[G] alpha;             // group-specific intercepts (6 cells)

  real beta_depth;
  real<lower=0> sigma;
}

model {
  mu_alpha  ~ normal(0, 1);
  tau_alpha ~ cauchy(0, 1);

  alpha ~ normal(mu_alpha, tau_alpha);

  beta_depth ~ normal(0, 1);
  sigma      ~ exponential(1);

  for (i in 1:N)
    bill_length[i] ~ normal(alpha[group[i]] + beta_depth * bill_depth[i], sigma);
}

generated quantities {
  vector[N] yrep;
  vector[N] log_lik;
  for (i in 1:N) {
    real mu_i  = alpha[group[i]] + beta_depth * bill_depth[i];
    yrep[i]    = normal_rng(mu_i, sigma);
    log_lik[i] = normal_lpdf(bill_length[i] | mu_i, sigma);
  }
}
