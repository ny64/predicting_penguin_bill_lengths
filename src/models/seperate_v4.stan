data {
  int<lower=1> N;              
  int<lower=1> J;            
  array[N] int<lower=1, upper=J> species;  
  vector[N] sex;
  vector[N] bill_length;         
  vector[N] bill_depth;           
}

parameters {
  vector[J] alpha;          // species-specific intercepts
  vector[J] beta_depth;     // species-specific slopes 
  vector[J] beta_sex;       // sex-specific slopes
  vector<lower=0>[J] sigma; // species-specific errors
}

model {
  // Priors
  alpha       ~ normal(0, 1);
  beta_depth  ~ normal(0, 1);
  beta_sex    ~ normal(0, 1);
  sigma       ~ normal(0, 1);

  // Likelihood
  vector[N] Mu ;
  vector[N] Sigma;
  for (n in 1:N) {
    Mu[n] = alpha[species[n]] + 
            beta_depth[species[n]] * bill_depth[n] +
            beta_sex[species[n]] * sex[n];
    Sigma[n] = sigma[species[n]];
  }
  bill_length ~ normal(Mu, Sigma);
}

generated quantities {
  vector[N] log_lik;
  vector[N] y_rep;

  for (n in 1:N) {
    real Mu = alpha[species[n]] + 
              beta_depth[species[n]] * bill_depth[n] +
              beta_sex[species[n]] * sex[n];
    real Sigma = sigma[species[n]];
    
    log_lik[n] = normal_lpdf(bill_length[n] | Mu, Sigma);
    y_rep[n]   = normal_rng(Mu, Sigma);
  }
}