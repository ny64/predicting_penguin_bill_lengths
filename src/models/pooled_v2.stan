// Complete pooling (spec-conforming): single intercept, no species terms.
// bill_length_i ~ Normal(alpha + beta1*x_i + beta2*sex_i, sigma)
data {
  int<lower=0> N;
  vector[N] bill_length;   // outcome (scaled)
  vector[N] bill_depth;    // predictor (centred+scaled)
  vector[N] sex;           // 0 = female, 1 = male
}

parameters {
  real alpha;
  real beta_depth;
  real beta_sex;
  real<lower=0> sigma;
}

model {
  alpha      ~ normal(0, 1);
  beta_depth ~ normal(0, 1);
  beta_sex   ~ normal(0, 1);
  sigma      ~ exponential(1);

  bill_length ~ normal(alpha + beta_depth * bill_depth + beta_sex * sex, sigma);
}

generated quantities {
  vector[N] yrep;
  vector[N] log_lik;
  for (i in 1:N) {
    real mu_i  = alpha + beta_depth * bill_depth[i] + beta_sex * sex[i];
    yrep[i]    = normal_rng(mu_i, sigma);
    log_lik[i] = normal_lpdf(bill_length[i] | mu_i, sigma);
  }
}
