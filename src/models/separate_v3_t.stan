data {
  int<lower=1> N;    
  int<lower=1> G;             
  vector[N] bill_length;          // outcome
  vector[N] bill_depth;           // predcictor
  array[N] int group;             // group indices 1...G
}

parameters {
  vector[G] alpha;             //intercepts
  vector[G] beta;              //slopes
  vector<lower=0>[G] sigma;    //errors
}

model {
  // priors
  alpha ~ student_t(3, 0, 1);
  beta  ~ student_t(3, 0, 1);
  sigma ~ exponential(1);


  // Likelihood
  for (n in 1:N) {
    real mu = alpha[group[n]] + beta[group[n]] * bill_depth[n];
    real Sigma = sigma[group[n]];
    bill_length[n] ~ normal(mu, Sigma);
  }

}

generated quantities {
  vector[N] log_lik;
  vector[N] yrep;

  for (n in 1:N) {
    real mu = alpha[group[n]] + beta[group[n]] * bill_depth[n];
    real Sigma = sigma[group[n]];
    
    yrep[n]    = normal_rng(mu, Sigma);
    log_lik[n] = normal_lpdf(bill_length[n] | mu, Sigma);
  }
}
