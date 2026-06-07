// Fully separated random slope model (no general intercept or slope)
// Target:    bill_length_mm
// Covariates: bill_depth_mm, with group structure defined by
//             species x sex combinations
// Variance:  Each group has its own residual SD that scales with bill_depth
//            (multiplicative heteroscedasticity: sigma_i = exp(a_sigma[g] + b_sigma[g] * bill_depth))

data {
  int<lower=1> N;           // total observations
  int<lower=1> G;           // number of groups (species x sex combinations)
  vector[N] bill_length;    // outcome (bill_length_mm)  
  vector[N] bill_depth;     // predictor (bill_depth_mm), NOT centered here;
                            // center before passing in for numerical stability
  array[N] int<lower=1, upper=G> group;  // group index for each observation
}

parameters {
  // Group-level intercepts and slopes (fully separated — no pooling, no global mean)
  vector[G] alpha;          // intercept per group
  vector[G] beta;           // slope per group

  // Heteroscedastic variance model: log(sigma) = a_sigma + b_sigma * bill_depth
  // Captures increasing variance with bill_depth, per group
  vector[G] a_sigma;        // log-scale baseline SD per group
  vector[G] b_sigma;        // log-scale slope of SD per group
}

transformed parameters {
  // Per-observation SD (must be positive by construction via exp)
  vector<lower=0>[N] sigma;
  for (n in 1:N) {
    sigma[n] = exp(a_sigma[group[n]] + b_sigma[group[n]] * bill_depth[n]);
  }
}

model {
  // ---- Priors ----
  // Weakly informative priors for intercepts and slopes
  alpha   ~ normal(0, 1);    // bill_length is roughly in 35–60 mm range
  beta    ~ normal(0, 1);      // slope of bill_length on bill_depth

  // Priors on log-SD model
  // a_sigma ~ N(0, 1) keeps baseline SD in a reasonable range (e^0 = 1 mm)
  a_sigma ~ normal(0, 1);
  // b_sigma near 0 but allows moderate heteroscedasticity
  b_sigma ~ normal(0, 0.5);

  // ---- Likelihood ----
  for (n in 1:N) {
    bill_length[n] ~ normal(alpha[group[n]] + beta[group[n]] * bill_depth[n],
                            sigma[n]);
  }
}

generated quantities {
  // Posterior predictive draws
  vector[N] yrep;

  // Pointwise log-likelihood (for loo / WAIC)
  vector[N] log_lik;

  for (n in 1:N) {
    real mu_n = alpha[group[n]] + beta[group[n]] * bill_depth[n];
    yrep[n]     = normal_rng(mu_n, sigma[n]);
    log_lik[n]  = normal_lpdf(bill_length[n] | mu_n, sigma[n]);
  }
}
