// =============================================================================
// Pooled Model v1 — Complete pooling (no species structure)
// =============================================================================
// What is pooled:   everything — single intercept, single slopes, single sigma
// What varies:      nothing by species; sex is modelled via a single dummy
//
// Species identity is ignored entirely. Instead, two dummy variables encode
// species membership as fixed effects (Adelie = baseline). This is the
// "complete pooling" extreme: all observations are treated as exchangeable
// except for the measured predictors bill_depth and sex.
//
// bill_length_i ~ Normal(alpha + beta_depth*x_i + beta_sex*sex_i
//                        + beta_species'*species_dummies_i, sigma)
// alpha          ~ Normal(0, 1)
// beta_depth     ~ Normal(0, 1)
// beta_sex       ~ Normal(0, 1)
// beta_species   ~ Normal(0, 1)   [2-vector for Chinstrap, Gentoo dummies]
// sigma          ~ Exponential(1)
// =============================================================================
data {
  int<lower=0> N;          // number of observations
  vector[N] bill_length;   // target (scaled)
  vector[N] bill_depth;    // predictor (scaled)
  vector[N] sex;           // dummy: 1 = male, 0 = female
  matrix[N, 2] species;    // 2 dummy variables for species
}

parameters {
  real<lower=-100, upper=100> alpha;              // intercept
  real<lower=-100, upper=100> beta_depth;         // bill depth effect
  real<lower=-100, upper=100> beta_sex;           // sex effect
  vector<lower=-100, upper=100>[2] beta_species;  // species effects
  real<lower=0, upper=100> sigma;     // noise
}

model {
  // Priors
  //alpha ~ uniform(-1, 1);
  //beta_depth ~ uniform(-1, 1);
  //beta_sex ~ uniform(-1, 1);
  //beta_species ~ uniform(-1, 1);
  //sigma ~ uniform(0, 2);
  
  // Likelihood
  bill_length ~ normal(alpha + 
                       beta_depth * bill_depth + 
                       beta_sex * sex + 
                       species * beta_species, sigma);
}

generated quantities {
  vector[N] yrep;
  vector[N] log_lik;
  for (i in 1:N) {
    real mu_i = alpha + beta_depth * bill_depth[i] + beta_sex * sex[i] +
                dot_product(species[i], beta_species);
    yrep[i]    = normal_rng(mu_i, sigma);
    log_lik[i] = normal_lpdf(bill_length[i] | mu_i, sigma);
  }
}
