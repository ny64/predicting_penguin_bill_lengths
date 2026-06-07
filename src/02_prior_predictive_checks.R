# PRIOR PREDICTIVE CHECKS
# Description: This file is to check our priors before subsequent modelling. In
#              this analysis we do not look at the data beforehand. To still 
#              create some realistic priors we suppose that we use scaled data
#              (with mean 0 and sd 1). In this file we re-scale to see if the 
#              generated data is realistic. Assumptions about beak length are:
#              normal distributed.
# Author: Peter Breitzler

set.seed(187)

penguins <- readRDS("data/penguins.RDS")
y <- penguins$bill_length
X <- penguins$bill_depth
y_scaled <- scale(y)  # (y - mean(y)) / sd(y)
X_scaled <- scale(X)  # (x - mean(x)) / sd(x)

# LIKELIHOOD
# Say our target (length) is normal distributed
# So we have mu and sigma as parameters
#
# PRIORs
# mu ~ N(0, 1)  we scale our data; this lets us have this non-informative type prior
# sigma ~ Exp(1)  common prior for variance

n_sims <- 1000

# Sample from priors
intercept <- rnorm(n_sims, mean=0, sd=1)
slope <- rnorm(n_sims, mean=0, sd=1)
sigma <- rexp(n_sims, rate=1)

mu_scaled <- intercept + slope * 0  # mean of X_scaled = 0

y_pred_scaled <- rnorm(n_sims, mean = mu_scaled, sd = sigma)

y_pred <- y_pred_scaled * sd(y) + mean(y)
png("report/figures/prior_pred_marginal.png", width = 800, height = 500)
hist(y_pred, main = "Prior Predictive Check", xlab = "Bill Length (mm)", breaks = 50)
abline(v = range(y), col = "red", lwd = 2, lty = 2)
dev.off()

# HIERARCHICAL PRIORS (hierarchical_v3)
# The hierarchical model introduces hyperpriors for the population-level means
# and SDs of the three species-varying coefficients (intercept, depth slope,
# sex effect). The species-level parameters are then drawn from those hyperpriors.
#
# HYPERPRIORS
# mu_alpha      ~ N(0, 1)   population mean of species intercepts
# mu_beta_depth ~ N(0, 1)   population mean of depth slopes
# mu_sex        ~ N(0, 1)   population mean of sex effects
# sigma_alpha   ~ Exp(1)    population SD of species intercepts
# sigma_beta    ~ Exp(1)    population SD of depth slopes
# sigma_sex     ~ Exp(1)    population SD of sex effects
#
# SPECIES-LEVEL PRIORS (drawn from hyperpriors)
# alpha_s     ~ N(mu_alpha,      sigma_alpha)   for s = 1,2,3
# beta_depth_s ~ N(mu_beta_depth, sigma_beta)   for s = 1,2,3
# beta_sex_s  ~ N(mu_sex,        sigma_sex)     for s = 1,2,3

J <- 3  # number of species

# ====================== Hyperprior checks =====================================

mu_alpha_prior      <- rnorm(n_sims, mean = 0, sd = 1)
mu_beta_depth_prior <- rnorm(n_sims, mean = 0, sd = 1)
mu_sex_prior        <- rnorm(n_sims, mean = 0, sd = 1)

# Re-scaled to mm for interpretability
hist(mu_alpha_prior * sd(y) + mean(y),
     main = "Prior: mu_alpha (population intercept mean)",
     xlab = "Bill Length (mm)", breaks = 50)
abline(v = range(y), col = "red", lwd = 2, lty = 2)

hist(mu_beta_depth_prior * sd(y),
     main = "Prior: mu_beta_depth (population depth slope mean)",
     xlab = "Change in bill length per SD of bill depth (mm)", breaks = 50)
abline(v = 0, col = "blue", lwd = 1, lty = 2)

hist(mu_sex_prior * sd(y),
     main = "Prior: mu_sex (population sex effect mean)",
     xlab = "Male - female bill length difference (mm)", breaks = 50)
abline(v = 0, col = "blue", lwd = 1, lty = 2)

# Exp(1) for the sigma hyperpriors: most prior mass below ~2 on z-score scale,
# meaning species are not expected to differ by more than ~2 SDs in their
# coefficients. Heavy tail allows larger differences if the data demands it.
sigma_alpha_prior <- rexp(n_sims, rate = 1)
sigma_beta_prior  <- rexp(n_sims, rate = 1)
sigma_sex_prior   <- rexp(n_sims, rate = 1)

hist(sigma_alpha_prior * sd(y),
     main = "Prior: sigma_alpha (species spread in intercepts)",
     xlab = "SD of species intercepts (mm)", breaks = 50)

hist(sigma_beta_prior * sd(y),
     main = "Prior: sigma_beta (species spread in depth slopes)",
     xlab = "SD of species depth slopes (mm per SD depth)", breaks = 50)

hist(sigma_sex_prior * sd(y),
     main = "Prior: sigma_sex (species spread in sex effects)",
     xlab = "SD of species sex effects (mm)", breaks = 50)

# ========================= Species-level prior draws =========================

alpha_s     <- matrix(NA, nrow = n_sims, ncol = J)
beta_depth_s <- matrix(NA, nrow = n_sims, ncol = J)
beta_sex_s  <- matrix(NA, nrow = n_sims, ncol = J)

for (s in 1:J) {
  alpha_s[, s]      <- rnorm(n_sims, mean = mu_alpha_prior,      sd = sigma_alpha_prior)
  beta_depth_s[, s] <- rnorm(n_sims, mean = mu_beta_depth_prior, sd = sigma_beta_prior)
  beta_sex_s[, s]   <- rnorm(n_sims, mean = mu_sex_prior,        sd = sigma_sex_prior)
}

sigma_hier <- rexp(n_sims, rate = 1)

# Prior predictive y for a female (sex=0) at mean depth (x=0), per species.
# With sex=0 and x=0, mu reduces to alpha_s alone — cleanest check of the
# intercept prior. Red lines mark the observed range of bill_length.
png("report/figures/prior_pred_species.png", width = 1200, height = 450)
par(mfrow = c(1, J))
species_names <- c("Adelie", "Chinstrap", "Gentoo")
for (s in 1:J) {
  mu_s   <- alpha_s[, s]  # x=0, sex=0
  y_s    <- rnorm(n_sims, mean = mu_s, sd = sigma_hier)
  y_s_mm <- y_s * sd(y) + mean(y)
  hist(y_s_mm,
       main  = paste("Prior predictive:", species_names[s]),
       xlab  = "Bill Length (mm)", breaks = 50)
  abline(v = range(y), col = "red", lwd = 2, lty = 2)
}
par(mfrow = c(1, 1))
dev.off()

# Prior predictive for a male (sex=1) vs female (sex=0) at mean depth,
# pooled across species. Shows whether the sex effect prior allows for
# plausible male-female differences without being too wide.
mu_female <- rowMeans(alpha_s)
mu_male   <- rowMeans(alpha_s) + rowMeans(beta_sex_s)

y_female_mm <- (rnorm(n_sims, mu_female, sigma_hier)) * sd(y) + mean(y)
y_male_mm   <- (rnorm(n_sims, mu_male,   sigma_hier)) * sd(y) + mean(y)

png("report/figures/prior_pred_sex_contrast.png", width = 800, height = 500)
hist(y_male_mm - y_female_mm,
     main = "Prior: male - female bill length (averaged over species)",
     xlab = "Difference (mm)", breaks = 50)
abline(v = 0, col = "blue", lwd = 1, lty = 2)
dev.off()

