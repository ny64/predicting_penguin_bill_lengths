// =============================================================================
// Hierarchical Model v6 (v3 Modified) — Pooled Sex Effect & Hierarchical Sigma
// =============================================================================
// What is hierarchical: alpha         (intercept per species, 3 groups)
//                       beta_depth    (bill-depth slope per species, 3 groups)
//                       sigma         (residual standard deviation per species, 3 groups)
//
// What is pooled:       beta_sex      (single additive shift for sex across all species)
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
  // 1. Hyperparameters for Means & Slopes
  real mu_alpha;
  real mu_beta_depth;
  real<lower=0> sigma_alpha;
  real<lower=0> sigma_beta;

  // 2. Hyperparameters for Variances (Hierarchical Sigma)
  real mu_log_sigma;
  real<lower=0> tau_log_sigma;

  // 3. Species-level parameters
  vector[J] alpha;             // Intercept per species
  vector[J] beta_depth;        // Depth slope per species
  vector<lower=0>[J] sigma;    // Variance per species

  // 4. Pooled Fixed Effect
  real beta_sex;               // Global male shift
}

model {
  // --- Hyperpriors ---
  mu_alpha      ~ normal(0, 1);
  mu_beta_depth ~ normal(0, 1);
  sigma_alpha   ~ exponential(1);
  sigma_beta    ~ exponential(1);

  // Hyperpriors for Hierarchical Variance
  // (Log-normal ensures standard deviations remain strictly positive)
  mu_log_sigma  ~ normal(0, 1);
  tau_log_sigma ~ exponential(1);

  // --- Species-Level Priors ---
  alpha      ~ normal(mu_alpha, sigma_alpha);
  beta_depth ~ normal(mu_beta_depth, sigma_beta);
  sigma      ~ lognormal(mu_log_sigma, tau_log_sigma);

  // --- Pooled Effect Prior ---
  beta_sex   ~ normal(0, 1);

  // --- Likelihood ---
  for (i in 1:N) {
    real mu_i = alpha[species[i]] +
                beta_depth[species[i]] * bill_depth[i] +
                beta_sex * sex[i];
    
    bill_length[i] ~ normal(mu_i, sigma[species[i]]);
  }
}

generated quantities {
  vector[N] yrep;
  vector[N] log_lik;
  
  for (i in 1:N) {
    real mu_i = alpha[species[i]] +
                beta_depth[species[i]] * bill_depth[i] +
                beta_sex * sex[i];
                
    yrep[i]    = normal_rng(mu_i, sigma[species[i]]);
    log_lik[i] = normal_lpdf(bill_length[i] | mu_i, sigma[species[i]]);
  }
}