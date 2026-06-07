// =============================================================================
// Hierarchical Model v1 — Species-varying intercepts AND depth slopes
// =============================================================================
// What is hierarchical: alpha_s     (intercept per species, 3 groups)
//                       beta_depth_s (bill-depth slope per species, 3 groups)
// What is pooled:       beta_sex    (single slope for sex)
//
// Compared to v2, this adds a species-specific slope for bill depth.
// Each species may have a different relationship between bill depth and length.
// Both alpha_s and beta_depth_s are drawn from their own shared distributions,
// so species still borrow strength from each other.
//
// bill_length_i ~ Normal(alpha_s[i] + beta_depth_s[i]*x_i + beta_sex*sex_i, sigma)
// alpha_s       ~ Normal(mu_alpha,      sigma_alpha)    [hierarchical]
// beta_depth_s  ~ Normal(mu_beta_depth, sigma_beta)     [hierarchical]
// mu_alpha      ~ Normal(0, 1)
// mu_beta_depth ~ Normal(0, 1)
// sigma_alpha   ~ Exponential(1)
// sigma_beta    ~ Exponential(1)
// beta_sex      ~ Normal(0, 1)                          [pooled]
// sigma         ~ Exponential(1)
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
  // Population level (hyperpriors)
  real mu_alpha;               // mean intercept across species
  real mu_beta_depth;          // mean bill depth effect across species
  real<lower=0> sigma_alpha;   // variation in intercepts across species
  real<lower=0> sigma_beta;    // variation in slopes across species
  
  // Species level
  vector[J] alpha;             // species specific intercepts
  vector[J] beta_depth;        // species specific bill depth slopes
  
  // Observation level
  real beta_sex;               // sex effect (pooled across species)
  real<lower=0> sigma;         // noise
}

model {
  // Hyperpriors
  mu_alpha      ~ normal(0, 1);
  mu_beta_depth ~ normal(0, 1);
  sigma_alpha   ~ exponential(1);
  sigma_beta    ~ exponential(1);
  
  // Species level priors (this is the hierarchical part)
  alpha       ~ student_t(3, mu_alpha, sigma_alpha);
  beta_depth  ~ student_t(3, mu_beta_depth, sigma_beta);
  
  // Observation level priors
  beta_sex ~ student_t(3, 0, 1);
  sigma    ~ exponential(1);
  
  // Likelihood
  for (i in 1:N) {
    bill_length[i] ~ normal(alpha[species[i]] + 
                            beta_depth[species[i]] * bill_depth[i] +
                            beta_sex * sex[i], sigma);
  }
}

generated quantities {
  vector[N] yrep;
  vector[N] log_lik;
  for (i in 1:N) {
    real mu_i = alpha[species[i]] + beta_depth[species[i]] * bill_depth[i] +
                beta_sex * sex[i];
    yrep[i]    = normal_rng(mu_i, sigma);
    log_lik[i] = normal_lpdf(bill_length[i] | mu_i, sigma);
  }
}

