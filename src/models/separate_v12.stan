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
  for (i in 1:N) {
    yrep[i] = normal_rng(alpha[species[i]] + 
                         beta_depth[species[i]] * bill_depth[i] +
                         beta_sex[species[i]] * sex[i], 
                         sigma[species[i]]);
  }
}

