data {
  int<lower=0> n; // observations
  int<lower=0> k; // number of models
  int gm; // grand mean
  int<lower=0> psigmu; // prior for mu_sigma
  int<lower=1, upper=8> x[n]; // group indicator
  vector[n] y; // outcome
}
parameters {
  real alpha[k]; // model means
  real<lower=0> sigma; // error
}
transformed parameters {
  real rf_xg_diff; // compare rf and xg
  real rf_ada_diff; // compare rf and ada
  real xg_ada_diff; // compare xg and ada
  rf_xg_diff = alpha[1] - alpha[2]; // compute diff
  rf_ada_diff = alpha[1] - alpha[3]; // compute diff
  xg_ada_diff = alpha[2] - alpha[3]; // compute diff
}
model {
  vector[n] mu; // expected value
  for (i in 1:n) { // for all people
    mu[i] = alpha[x[i]]; //their mu is their group's mean
  }
  y ~ normal(mu, sigma); // likelihood
  alpha ~ normal(gm, psigmu); // prior, defined from data
  sigma ~ cauchy(0, 1); // uninformative prior on sigma
}
