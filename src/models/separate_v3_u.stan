data {
  int<lower=1> N;    
  int<lower=1> G;             
  vector[N] bill_length;          // outcome
  vector[N] bill_depth;           // predcictor
  array[N] int group;             // group indices 1...G
}

parameters {
  vector<lower=-100, upper=100>[G] alpha;             //intercepts
  vector<lower=-100, upper=100>[G] beta;              //slopes
  vector<lower=0, upper=100>[G] sigma;    //errors
}

model {
  // priors
  // alpha ~ uniform(-1, 1);
  // beta  ~ uniform(-1, 1);
  // sigma ~ uniform(0, 2);


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
