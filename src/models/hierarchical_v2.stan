// =============================================================================
// Hierarchical Model v2 — Species-varying intercepts only
// =============================================================================
// What is hierarchical: alpha_s (intercept per species, 3 groups)
// What is pooled:       beta_depth (single slope for bill depth)
//                       beta_sex   (single slope for sex)
//
// Each species gets its own baseline bill length (alpha_s), drawn from a
// shared Normal(mu_alpha, tau_alpha). The effect of bill depth and sex is
// assumed identical across all species — only the mean level differs.
//
// This is the minimal hierarchical extension of the pooled model:
// one new degree of freedom per species, no species-specific slope variation.
//
// bill_length_i ~ Normal(alpha_s[i] + beta_depth*x_i + beta_sex*sex_i, sigma)
// alpha_s       ~ Normal(mu_alpha, tau_alpha)    [hierarchical]
// mu_alpha      ~ Normal(0, 1)
// tau_alpha     ~ half-Cauchy(0, 1)
// beta_depth    ~ Normal(0, 1)                   [pooled]
// beta_sex      ~ Normal(0, 1)                   [pooled]
// sigma         ~ Exponential(1)
// =============================================================================
data {
  int<lower=0> N;
  int<lower=0> J;              // number of species (3)
  vector[N] bill_length;       // outcome (scaled)
  vector[N] bill_depth;        // predictor (centred+scaled)
  vector[N] sex;               // 0 = female, 1 = male
  array[N] int species;        // species index 1..J
}

parameters {
  real mu_alpha;               // population mean intercept
  real<lower=0> tau_alpha;     // population SD of intercepts (half-Cauchy)
  vector[J] alpha;             // species-specific intercepts

  real beta_depth;             // single pooled slope
  real beta_sex;
  real<lower=0> sigma;
}

model {
  // Hyperpriors
  mu_alpha  ~ normal(0, 1);
  tau_alpha ~ cauchy(0, 1);    // half-Cauchy (lower=0 in parameters block)

  // Species-level
  alpha ~ normal(mu_alpha, tau_alpha);

  // Observation-level
  beta_depth ~ normal(0, 1);
  beta_sex   ~ normal(0, 1);
  sigma      ~ exponential(1);

  for (i in 1:N)
    bill_length[i] ~ normal(alpha[species[i]] + beta_depth * bill_depth[i] +
                            beta_sex * sex[i], sigma);
}

generated quantities {
  vector[N] yrep;
  vector[N] log_lik;
  for (i in 1:N) {
    real mu_i  = alpha[species[i]] + beta_depth * bill_depth[i] + beta_sex * sex[i];
    yrep[i]    = normal_rng(mu_i, sigma);
    log_lik[i] = normal_lpdf(bill_length[i] | mu_i, sigma);
  }
}
