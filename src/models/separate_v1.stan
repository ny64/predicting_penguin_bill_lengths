// =============================================================================
// Separate Model v1 — Fully independent parameters per species
// =============================================================================
// What varies by species: alpha_s      (intercept, 3 groups)
//                         beta_depth_s  (bill-depth slope, 3 groups)
//                         beta_sex_s    (sex effect, 3 groups)
//                         sigma_s       (residual SD, 3 groups)
// What is shared:         nothing — no pooling across species whatsoever
//
// Each species is fit completely independently. There are no hyperpriors and
// no information sharing between groups. This is the "no pooling" extreme,
// contrasted with the complete-pooling model (pooled_v1) and the hierarchical
// model (hierarchical_v3) which partially pools via hyperpriors.
//
// bill_length_i ~ Normal(alpha_s[i] + beta_depth_s[i]*x_i + beta_sex_s[i]*sex_i, sigma_s[i])
// alpha_s       ~ Normal(0, 1)   [independent per species]
// beta_depth_s  ~ Normal(0, 1)   [independent per species]
// beta_sex_s    ~ Normal(0, 1)   [independent per species]
// sigma_s       ~ Exponential(1) [independent per species]
// =============================================================================
data {
  int<lower=0> N;              // number of observations
  int<lower=0> J;              // number of species (3)
  vector[N] bill_length;       // target (scaled)
  vector[N] bill_depth;        // predictor (scaled)
  vector[N] sex;               // dummy: 1 = male, 0 = female
  array[N] int species;        // species index (1, 2, or 3)
}

parameters {
  // Each species gets completely independent parameters
  vector[J] alpha;             // species specific intercepts
  vector[J] beta_depth;        // species specific bill depth slopes
  vector[J] beta_sex;          // species specific sex effects
  vector<lower=0>[J] sigma;    // species specific noise levels
}

model {
  // Independent priors for each species (no sharing!)
  alpha      ~ normal(0, 1);
  beta_depth ~ normal(0, 1);
  beta_sex   ~ normal(0, 1);
  sigma      ~ exponential(1);
  
  // Likelihood
  for (i in 1:N) {
    bill_length[i] ~ normal(alpha[species[i]] + 
                            beta_depth[species[i]] * bill_depth[i] +
                            beta_sex[species[i]] * sex[i], 
                            sigma[species[i]]);
  }
}

generated quantities {
  vector[N] yrep;
  vector[N] log_lik;
  for (i in 1:N) {
    real mu_i = alpha[species[i]] + beta_depth[species[i]] * bill_depth[i] +
                beta_sex[species[i]] * sex[i];
    yrep[i]    = normal_rng(mu_i, sigma[species[i]]);
    log_lik[i] = normal_lpdf(bill_length[i] | mu_i, sigma[species[i]]);
  }
}

