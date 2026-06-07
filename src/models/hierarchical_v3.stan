// =============================================================================
// Hierarchical Model v3 — Species-varying intercepts, depth slopes, AND sex slopes
// =============================================================================
// What is hierarchical: alpha_s      (intercept per species, 3 groups)
//                       beta_depth_s  (bill-depth slope per species, 3 groups)
//                       beta_sex_s    (sex effect per species, 3 groups)
// What is pooled:       nothing — all coefficients vary by species
//
// Compared to v1, this adds a species-specific sex effect. The male/female
// difference in bill length is no longer assumed equal across species; instead
// each species has its own beta_sex_s drawn from a shared Normal(mu_sex, sigma_sex).
// This allows the model to learn, e.g., that Gentoo shows a larger sex dimorphism
// than Chinstrap, while still pooling information across species.
//
// bill_length_i ~ Normal(alpha_s[i] + beta_depth_s[i]*x_i + beta_sex_s[i]*sex_i, sigma)
// alpha_s       ~ Normal(mu_alpha,      sigma_alpha)    [hierarchical]
// beta_depth_s  ~ Normal(mu_beta_depth, sigma_beta)     [hierarchical]
// beta_sex_s    ~ Normal(mu_sex,        sigma_sex)      [hierarchical]
// mu_alpha      ~ Normal(0, 1)
// mu_beta_depth ~ Normal(0, 1)
// mu_sex        ~ Normal(0, 1)
// sigma_alpha   ~ Exponential(1)
// sigma_beta    ~ Exponential(1)
// sigma_sex     ~ Exponential(1)
// sigma         ~ Exponential(1)
// =============================================================================
data {
  int<lower=0> N;
  int<lower=0> J;              // number of species (3)
  vector[N] bill_length;
  vector[N] bill_depth;
  vector[N] sex;               // 0 = female, 1 = male
  array[N] int species;        // species index 1..J
}

parameters {
  // Hyperparameters
  real mu_alpha;
  real mu_beta_depth;
  real mu_sex;
  real<lower=0> sigma_alpha;
  real<lower=0> sigma_beta;
  real<lower=0> sigma_sex;

  // Species-level coefficients
  vector[J] alpha;
  vector[J] beta_depth;
  vector[J] beta_sex;

  real<lower=0> sigma;
}

model {
  // Hyperpriors
  mu_alpha      ~ normal(0, 1);
  mu_beta_depth ~ normal(0, 1);
  mu_sex        ~ normal(0, 1);
  sigma_alpha   ~ exponential(1);
  sigma_beta    ~ exponential(1);
  sigma_sex     ~ exponential(1);

  // Species-level priors
  alpha      ~ normal(mu_alpha,      sigma_alpha);
  beta_depth ~ normal(mu_beta_depth, sigma_beta);
  beta_sex   ~ normal(mu_sex,        sigma_sex);

  sigma ~ exponential(1);

  for (i in 1:N)
    bill_length[i] ~ normal(alpha[species[i]] +
                            beta_depth[species[i]] * bill_depth[i] +
                            beta_sex[species[i]]   * sex[i], sigma);
}

generated quantities {
  vector[N] yrep;
  vector[N] log_lik;
  for (i in 1:N) {
    real mu_i  = alpha[species[i]] +
                 beta_depth[species[i]] * bill_depth[i] +
                 beta_sex[species[i]]   * sex[i];
    yrep[i]    = normal_rng(mu_i, sigma);
    log_lik[i] = normal_lpdf(bill_length[i] | mu_i, sigma);
  }
}
