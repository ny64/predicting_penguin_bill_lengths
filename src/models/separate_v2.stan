data {
  int<lower=1>                        N;
  int<lower=1>                        n_species;    // = 3
  int<lower=1>                        n_sex;        // = 2
  array[N] int<lower=1, upper=n_species> species;
  array[N] int<lower=1, upper=n_sex>    sex;
  vector[N]                           bill_depth;
  vector[N]                           bill_length;
}

parameters {
  // Each group gets its own intercept — completely independent priors
  vector[n_species]  alpha_species;   // one intercept per species
  vector[n_sex]      alpha_sex;       // one intercept per sex
  vector<lower=0>[n_species]     beta_species;
  vector<lower=0>[n_sex]         beta_sex;
  real<lower=0>                  sigma;
  real<lower=0>      nu;
}

model {

  alpha_species ~ normal(0, 1);
  alpha_sex     ~ normal(0, 1);
  beta_species  ~ normal(0, 1);
  beta_sex      ~ normal(0, 1);

  nu            ~ gamma(2, .1);
  sigma         ~ exponential(1);

  {
    vector[N] mu;
    for (n in 1:N)
        mu[n] = alpha_species[species[n]]
              + alpha_sex[sex[n]]
              + (beta_species[species[n]] + beta_sex[sex[n]]) * bill_depth[n];
    bill_length ~ student_t(nu, mu, sigma);
  }

}

generated quantities {
  vector[N] y_rep;
  vector[N] log_lik;

  for (n in 1:N) {
    real mu_n = alpha_species[species[n]]
                + alpha_sex[sex[n]]
                + (beta_species[species[n]] + beta_sex[sex[n]]) * bill_depth[n];
    y_rep[n]   = student_t_rng(nu, mu_n, sigma);
    log_lik[n] = student_t_lpdf(bill_length[n] | nu, mu_n, sigma);
  }
}