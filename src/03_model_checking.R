# =============================================================================
# Model checking — posterior predictive checks for all four models
# =============================================================================
# This file runs posterior predictive checks (PPCs) for:
#   1. pooled_v1         — complete pooling, species as fixed-effect dummies
#   2. hierarchical_v1   — partial pooling, intercepts + depth slopes vary by
#                          species; sex effect is a single global coefficient
#   3. hierarchical_v3   — partial pooling, ALL coefficients (incl. sex) vary
#                          by species
#   4. separate_v1       — no pooling, fully independent parameters per species
# =============================================================================

library(rstan)
library(bayesplot)
library(posterior)
library(tidyverse)

options(mc.cores = parallelly::availableCores())
rstan_options(auto_write = TRUE)

theme_set(bayesplot::theme_default(base_family = "sans"))

# -----------------------------------------------------------------------------
# Data
# -----------------------------------------------------------------------------

penguins <- readRDS("data/penguins.RDS")

penguins_clean <- penguins %>%
  drop_na(bill_length, bill_depth, sex, species)

bill_length_scaled <- scale(penguins_clean$bill_length)[, 1]
bill_depth_scaled  <- scale(penguins_clean$bill_depth)[, 1]
sex_dummy          <- as.integer(penguins_clean$sex == "male")
species_index      <- as.integer(penguins_clean$species)
# pooled_v1 encodes species as two dummy columns (Adelie = reference)
species_dummy      <- model.matrix(~ species - 1, data = penguins_clean)[, -1]
# special for separated model
group_index        <- mutate(penguins, groups = interaction(species, sex))$groups |> as.integer()

y                  <- bill_length_scaled
species_labels     <- as.character(penguins_clean$species)
sex_labels         <- as.character(penguins_clean$sex)
species_sex_labels <- paste(penguins_clean$species, penguins_clean$sex, sep = "-")
group_labels       <- mutate(penguins, groups = interaction(species, sex))$groups |> as.character()

stan_data_pooled <- list(
  N           = nrow(penguins_clean),
  bill_length = bill_length_scaled,
  bill_depth  = bill_depth_scaled,
  sex         = sex_dummy,
  species     = species_dummy
)

stan_data_separate <- list(
  N           = nrow(penguins_clean),
  G           = 6L,
  bill_length = bill_length_scaled,
  bill_depth  = bill_depth_scaled,
  group         = group_index
)

stan_data_indexed <- list(
  N           = nrow(penguins_clean),
  J           = 3L,
  bill_length = bill_length_scaled,
  bill_depth  = bill_depth_scaled,
  sex         = sex_dummy,
  species     = species_index
)

# -----------------------------------------------------------------------------
# Fit / load models
# -----------------------------------------------------------------------------
# Set REFIT_MODELS <- TRUE to re-run MCMC from scratch.
# Set REFIT_MODELS <- FALSE to load previously saved .RDS fits from src/models/.
# Set SAVE_FITS <- TRUE to write fits to disk after sampling (ignored when loading).

REFIT_MODELS <- FALSE
SAVE_FITS    <- TRUE

rds_pooled      <- "src/models/fit_pooled_v1.RDS"
rds_hier_v1     <- "src/models/fit_hierarchical_v1.RDS"
rds_hier_v3     <- "src/models/fit_hierarchical_v3.RDS"
rds_separate    <- "src/models/fit_separate_v3.RDS"

if (REFIT_MODELS) {
  fit_pooled <- stan(
    file   = "src/models/pooled_v1.stan",
    data   = stan_data_pooled,
    chains = 4, iter = 2000, warmup = 1000, seed = 187
  )
  fit_hier_v1 <- stan(
    file   = "src/models/hierarchical_v1.stan",
    data   = stan_data_indexed,
    chains = 4, iter = 2000, warmup = 1000, seed = 187
  )
  fit_hier_v3 <- stan(
    file   = "src/models/hierarchical_v3.stan",
    data   = stan_data_indexed,
    chains = 4, iter = 2000, warmup = 1000, seed = 187
  )
  fit_separate <- stan(
    file   = "src/models/separate_v3.stan",
    data   = stan_data_separate,
    chains = 4, iter = 2000, warmup = 1000, seed = 187
  )

  if (SAVE_FITS) {
    saveRDS(fit_pooled,   rds_pooled)
    saveRDS(fit_hier_v1,  rds_hier_v1)
    saveRDS(fit_hier_v3,  rds_hier_v3)
    saveRDS(fit_separate, rds_separate)
    message("Fits saved to src/models/")
  }
} else {
  fit_pooled   <- readRDS(rds_pooled)
  fit_hier_v1  <- readRDS(rds_hier_v1)
  fit_hier_v3  <- readRDS(rds_hier_v3)
  fit_separate <- readRDS(rds_separate)
  message("Fits loaded from src/models/")
}

# -----------------------------------------------------------------------------
# Extract yrep draws
# -----------------------------------------------------------------------------

yrep_pooled   <- as_draws_matrix(fit_pooled)   %>% subset_draws(variable = "yrep")
yrep_hier_v1  <- as_draws_matrix(fit_hier_v1)  %>% subset_draws(variable = "yrep")
yrep_hier_v3  <- as_draws_matrix(fit_hier_v3)  %>% subset_draws(variable = "yrep")
yrep_separate <- as_draws_matrix(fit_separate) %>% subset_draws(variable = "yrep")

# =================== PLOTS USED IN THE REPORT =================================

library(patchwork)

p1 <- ppc_dens_overlay(y, yrep_pooled[1:num_rep, ]) +
  ggtitle("Pooled")

p2 <- ppc_dens_overlay(y, yrep_hier_v1[1:num_rep, ]) +
  ggtitle("Hierarchical (pooled sex)")

p3 <- ppc_dens_overlay(y, yrep_hier_v3[1:num_rep, ]) +
  ggtitle("Hierarchical (varying sex)")

p4 <- ppc_dens_overlay(y, yrep_separate[1:num_rep, ]) +
  ggtitle("Separate")

(p1 + p2) / (p3 + p4) +
  plot_annotation(
    caption = "Predictive density overlays for all four models. The dark line is the observed\nbill_length distribution; light lines are 100 draws from the posterior predictive distribution."
  )

# Mean
p1_mean <- ppc_stat(y, yrep_pooled, stat = "mean") + ggtitle("Pooled")
p2_mean <- ppc_stat(y, yrep_hier_v1, stat = "mean") + ggtitle("Hierarchical (pooled sex)")
p3_mean <- ppc_stat(y, yrep_hier_v3, stat = "mean") + ggtitle("Hierarchical (varying sex)")
p4_mean <- ppc_stat(y, yrep_separate, stat = "mean") + ggtitle("Separate")

(p1_mean + p2_mean) / (p3_mean + p4_mean) +
  plot_annotation(
    caption = "Posterior predictive checks for the mean across all four models. The dark line is the\nobserved mean of bill_length; the histogram shows the distribution of means from posterior predictive draws."
  )

# SD
p1_sd <- ppc_stat(y, yrep_pooled, stat = "sd") + ggtitle("Pooled")
p2_sd <- ppc_stat(y, yrep_hier_v1, stat = "sd") + ggtitle("Hierarchical (pooled sex)")
p3_sd <- ppc_stat(y, yrep_hier_v3, stat = "sd") + ggtitle("Hierarchical (varying sex)")
p4_sd <- ppc_stat(y, yrep_separate, stat = "sd") + ggtitle("Separate")

(p1_sd + p2_sd) / (p3_sd + p4_sd) +
  plot_annotation(
    caption = "Posterior predictive checks for the standard deviation across all four models. The dark line is the\nobserved SD of bill_length; the histogram shows the distribution of SDs from posterior predictive draws."
  )

# 2D Min/Max
p1_mm <- ppc_stat_2d(y, yrep_pooled, stat = c("min", "max")) + ggtitle("Pooled")
p2_mm <- ppc_stat_2d(y, yrep_hier_v1, stat = c("min", "max")) + ggtitle("Hierarchical (pooled sex)")
p3_mm <- ppc_stat_2d(y, yrep_hier_v3, stat = c("min", "max")) + ggtitle("Hierarchical (varying sex)")
p4_mm <- ppc_stat_2d(y, yrep_separate, stat = c("min", "max")) + ggtitle("Separate")

(p1_mm + p2_mm) / (p3_mm + p4_mm) +
  plot_annotation(
    caption = "Posterior predictive checks for the min and max across all four models. The dark point is the\nobserved (min, max) of bill_length; the scatter shows (min, max) pairs from posterior predictive draws."
  )

# =============================================================================
# A. Global checks
# =============================================================================

ppc_hist(y, yrep_pooled[1:8, ]) +
  ggtitle("Histogram check: pooled_v1")

ppc_hist(y, yrep_hier_v1[1:8, ]) +
  ggtitle("Histogram check: hierarchical_v1 (pooled sex)")

ppc_hist(y, yrep_hier_v3[1:8, ]) +
  ggtitle("Histogram check: hierarchical_v3 (varying sex)")

ppc_hist(y, yrep_separate[1:8, ]) +
  ggtitle("Histogram check: separate_v1")
# --- A2. Density overlay: y vs 100 yrep samples ------------------------------
# Thick line is observed; thin lines are posterior predictive draws.

num_rep <- 100

ppc_dens_overlay(y, yrep_pooled[1:num_rep, ]) +
  ggtitle("Density overlay: pooled_v1")

ppc_dens_overlay(y, yrep_hier_v1[1:num_rep, ]) +
  ggtitle("Density overlay: hierarchical_v1 (pooled sex)")

ppc_dens_overlay(y, yrep_hier_v3[1:num_rep, ]) +
  ggtitle("Density overlay: hierarchical_v3 (varying sex)")

ppc_dens_overlay(y, yrep_separate[1:num_rep, ]) +
  ggtitle("Density overlay: separate_v1")

# --- A3. ECDF overlay --------------------------------------------------------
# Cumulative version of the density check; tail deviations are easier to read.

num_rep <- 10

ppc_ecdf_overlay(y, yrep_pooled[1:num_rep, ]) +
  ggtitle("ECDF overlay: pooled_v1")

ppc_ecdf_overlay(y, yrep_hier_v1[1:num_rep, ]) +
  ggtitle("ECDF overlay: hierarchical_v1 (pooled sex)")

ppc_ecdf_overlay(y, yrep_hier_v3[1:num_rep, ]) +
  ggtitle("ECDF overlay: hierarchical_v3 (varying sex)")

ppc_ecdf_overlay(y, yrep_separate[1:num_rep, ]) +
  ggtitle("ECDF overlay: separate_v1")

# --- A4. Test statistics: min and max ----------------------------------------
# T(y) (vertical line) should sit near the centre of the T(y_rep) histogram.
# Min/max probe the tails, which the mean parameter cannot absorb directly.

ppc_stat(y, yrep_pooled,   stat = "min") + ggtitle("Min statistic: pooled_v1")
ppc_stat(y, yrep_hier_v1,  stat = "min") + ggtitle("Min statistic: hierarchical_v1")
ppc_stat(y, yrep_hier_v3,  stat = "min") + ggtitle("Min statistic: hierarchical_v3")
ppc_stat(y, yrep_separate, stat = "min") + ggtitle("Min statistic: separate_v1")

ppc_stat(y, yrep_pooled,   stat = "max") + ggtitle("Max statistic: pooled_v1")
ppc_stat(y, yrep_hier_v1,  stat = "max") + ggtitle("Max statistic: hierarchical_v1")
ppc_stat(y, yrep_hier_v3,  stat = "max") + ggtitle("Max statistic: hierarchical_v3")
ppc_stat(y, yrep_separate, stat = "max") + ggtitle("Max statistic: separate_v1")

# --- A5. Joint min/max check -------------------------------------------------
# 2D joint distribution of (min, max) across replicates. The dot is observed.

color_scheme_set("brewer-Paired")

ppc_stat_2d(y, yrep_pooled,   stat = c("min", "max")) + ggtitle("min/max joint: pooled_v1")
ppc_stat_2d(y, yrep_hier_v1,  stat = c("min", "max")) + ggtitle("min/max joint: hierarchical_v1")
ppc_stat_2d(y, yrep_hier_v3,  stat = c("min", "max")) + ggtitle("min/max joint: hierarchical_v3")
ppc_stat_2d(y, yrep_separate, stat = c("min", "max")) + ggtitle("min/max joint: separate_v1")

color_scheme_set()

# --- A6. Standard deviation as test statistic --------------------------------
# SD is not directly constrained by a model parameter, so it is a genuine test.
# separate_v1 uses per-species sigmas; a shared single sigma in the hierarchical
# models could produce under- or over-dispersion if heteroscedasticity is large.

ppc_stat(y, yrep_pooled,   stat = "sd") + ggtitle("SD statistic: pooled_v1")
ppc_stat(y, yrep_hier_v1,  stat = "sd") + ggtitle("SD statistic: hierarchical_v1")
ppc_stat(y, yrep_hier_v3,  stat = "sd") + ggtitle("SD statistic: hierarchical_v3")
ppc_stat(y, yrep_separate, stat = "sd") + ggtitle("SD statistic: separate_v1")

# =============================================================================
# B. Species-grouped checks
# =============================================================================
# Core test for pooled_v1: it uses a single shared depth slope for all species.
# If Gentoo's depth-length slope differs from Adelie's, pooled_v1 will show
# systematic within-species residuals. Both hierarchical models (v1, v3) and
# separate_v1 allow species-specific depth slopes, so they should all perform
# similarly here. Any remaining gap between hier_v1 and hier_v3 in these panels
# would reflect their different sex specifications spilling into species means.

# --- B1. Density by species --------------------------------------------------

ppc_dens_overlay_grouped(y, yrep_pooled[1:100, ],   group = species_labels) +
  ggtitle("Species density: pooled_v1")

ppc_dens_overlay_grouped(y, yrep_hier_v1[1:100, ],  group = species_labels) +
  ggtitle("Species density: hierarchical_v1 (pooled sex)")

ppc_dens_overlay_grouped(y, yrep_hier_v3[1:100, ],  group = species_labels) +
  ggtitle("Species density: hierarchical_v3 (varying sex)")

ppc_dens_overlay_grouped(y, yrep_separate[1:100, ], group = group_labels) +
  ggtitle("Species density: separate_v1")

# --- B2. Mean by species -----------------------------------------------------

ppc_stat_grouped(y, yrep_pooled,   group = species_labels, stat = "mean") +
  ggtitle("Species mean: pooled_v1")

ppc_stat_grouped(y, yrep_hier_v1,  group = species_labels, stat = "mean") +
  ggtitle("Species mean: hierarchical_v1 (pooled sex)")

ppc_stat_grouped(y, yrep_hier_v3,  group = species_labels, stat = "mean") +
  ggtitle("Species mean: hierarchical_v3 (varying sex)")

ppc_stat_grouped(y, yrep_separate, group = species_labels, stat = "mean") +
  ggtitle("Species mean: separate_v1")

# --- B3. SD by species -------------------------------------------------------
# A shared sigma forces within-species variability to be equal across models
# (except separate_v1 which has per-species sigma). If SD checks fail for both
# hierarchical models but pass for separate_v1, per-species sigma is warranted.

ppc_stat_grouped(y, yrep_pooled,   group = species_labels, stat = "sd") +
  ggtitle("Species SD: pooled_v1")

ppc_stat_grouped(y, yrep_hier_v1,  group = species_labels, stat = "sd") +
  ggtitle("Species SD: hierarchical_v1 (pooled sex)")

ppc_stat_grouped(y, yrep_hier_v3,  group = species_labels, stat = "sd") +
  ggtitle("Species SD: hierarchical_v3 (varying sex)")

ppc_stat_grouped(y, yrep_separate, group = species_labels, stat = "sd") +
  ggtitle("Species SD: separate_v1")

# =============================================================================
# C. Sex-grouped checks
# =============================================================================
# All four models include a sex effect, but hier_v1 uses a single global
# beta_sex while hier_v3 allows it to vary by species. The marginal sex checks
# here (collapsing across species) are the easy test — all models that include
# any sex term should pass. The sharper test is in section D (species×sex).

ppc_dens_overlay_grouped(y, yrep_pooled[1:100, ],   group = sex_labels) +
  ggtitle("Sex density: pooled_v1")

ppc_dens_overlay_grouped(y, yrep_hier_v1[1:100, ],  group = sex_labels) +
  ggtitle("Sex density: hierarchical_v1 (pooled sex)")

ppc_dens_overlay_grouped(y, yrep_hier_v3[1:100, ],  group = sex_labels) +
  ggtitle("Sex density: hierarchical_v3 (varying sex)")

ppc_dens_overlay_grouped(y, yrep_separate[1:100, ], group = sex_labels) +
  ggtitle("Sex density: separate_v1")

ppc_stat_grouped(y, yrep_pooled,   group = sex_labels, stat = "mean") +
  ggtitle("Sex mean: pooled_v1")

ppc_stat_grouped(y, yrep_hier_v1,  group = sex_labels, stat = "mean") +
  ggtitle("Sex mean: hierarchical_v1 (pooled sex)")

ppc_stat_grouped(y, yrep_hier_v3,  group = sex_labels, stat = "mean") +
  ggtitle("Sex mean: hierarchical_v3 (varying sex)")

ppc_stat_grouped(y, yrep_separate, group = sex_labels, stat = "mean") +
  ggtitle("Sex mean: separate_v1")

# =============================================================================
# D. Species×sex grouped checks
# =============================================================================
# The finest-grained test: 6 cells (3 species × 2 sexes), n ~30–80 per cell.
# This directly tests whether the sex dimorphism varies by species, which is
# precisely what distinguishes hier_v1 (single beta_sex) from hier_v3
# (species-specific beta_sex_s drawn from a shared hyperprior).
#
# If hier_v1 fits these cell means as well as hier_v3, the species-varying sex
# effect is not needed — a simpler model is preferred. If hier_v3 fits
# noticeably better in the cells where dimorphism is largest (Gentoo in
# particular), that justifies the additional complexity in v3.

ppc_stat_grouped(y, yrep_pooled,   group = species_sex_labels, stat = "mean") +
  ggtitle("Species×sex mean: pooled_v1")

ppc_stat_grouped(y, yrep_hier_v1,  group = species_sex_labels, stat = "mean") +
  ggtitle("Species×sex mean: hierarchical_v1 (pooled sex)")

ppc_stat_grouped(y, yrep_hier_v3,  group = species_sex_labels, stat = "mean") +
  ggtitle("Species×sex mean: hierarchical_v3 (varying sex)")

ppc_stat_grouped(y, yrep_separate, group = species_sex_labels, stat = "mean") +
  ggtitle("Species×sex mean: separate_v1")

ppc_dens_overlay_grouped(y, yrep_pooled[1:100, ],   group = species_sex_labels) +
  ggtitle("Species×sex density: pooled_v1")

ppc_dens_overlay_grouped(y, yrep_hier_v1[1:100, ],  group = species_sex_labels) +
  ggtitle("Species×sex density: hierarchical_v1 (pooled sex)")

ppc_dens_overlay_grouped(y, yrep_hier_v3[1:100, ],  group = species_sex_labels) +
  ggtitle("Species×sex density: hierarchical_v3 (varying sex)")

ppc_dens_overlay_grouped(y, yrep_separate[1:100, ], group = group_labels) +
  ggtitle("Species×sex density: separate_v1")

# =============================================================================
# E. Posterior parameter inspection — all four models
# =============================================================================
# The PPC checks above test whether each model reproduces the data; this
# section inspects what each model actually learned.
# Comparing posteriors across models shows where the four approaches agree
# and where they diverge due to different pooling assumptions.

species_labs <- c("[1]" = "Adelie", "[2]" = "Chinstrap", "[3]" = "Gentoo")
sep_species_labs <- c("[1]" = "Adelie.female", "[2]" = "Chinstrap.female",
                      "[3]" = "Gentoo.female", "[4]" = "Adelie.male",
                      "[5]" = "Chinstrap.male", "[6]" = "Gentoo.male")

draws_p  <- as_draws_matrix(fit_pooled)
draws_h1 <- as_draws_matrix(fit_hier_v1)
draws_h3 <- as_draws_matrix(fit_hier_v3)
draws_s  <- as_draws_matrix(fit_separate)

# --- E1. Shared scalar parameters: pooled_v1 ---------------------------------
# A single depth slope shared across all species is pooled_v1's key assumption.
# beta_depth here is a blend of three genuinely different slopes — potentially
# biased for every species.

mcmc_areas(
  subset_draws(draws_p, variable = c("alpha", "beta_depth", "beta_sex",
                                     "beta_species[1]", "beta_species[2]", "sigma")),
  prob = 0.89, point_est = "median"
) +
  labs(title    = "Pooled parameters: pooled_v1",
       subtitle = "Single depth slope and sex effect shared across all species")

# --- E2. Species-specific parameters: separate_v1 ----------------------------
# No pooling: each group has completely independent parameters.
# Wide intervals for Chinstrap (n=68) reveal the cost of no information sharing.
# Comparing these to hierarchical posteriors (E3, E4) shows how much the
# hyperprior regularises.

mcmc_areas(
  subset_draws(draws_s, variable = c("alpha[1]", "alpha[2]", "alpha[3]",
                                     "alpha[4]", "alpha[5]", "alpha[6]")),
  prob = 0.89, point_est = "median"
) +
  scale_y_discrete(labels = sep_species_labs) +
  labs(title    = "Species intercepts: separate_v1",
       subtitle = "Independent — no information shared across species")

mcmc_areas(
  subset_draws(draws_s, variable = c("beta[1]", "beta[2]", "beta[3]",
                                     "beta[4]", "beta[5]", "beta[6]")),
  prob = 0.89, point_est = "median"
) +
  scale_y_discrete(labels = sep_species_labs) +
  labs(title    = "Species depth slopes: separate_v1",
       subtitle = "Wide CIs for Chinstrap (n=68) expose the variance cost of no pooling")

mcmc_areas(
  subset_draws(draws_s, variable = c("beta_sex[1]", "beta_sex[2]", "beta_sex[3]")),
  prob = 0.89, point_est = "median"
) +
  scale_y_discrete(labels = species_labs) +
  labs(title    = "Species sex effects: separate_v1")

mcmc_areas(
  subset_draws(draws_s, variable = c("sigma[1]", "sigma[2]", "sigma[3]",
                                     "sigma[4]", "sigma[5]", "sigma[6]")),
  prob = 0.89, point_est = "median"
) +
  scale_y_discrete(labels = sep_species_labs) +
  labs(title    = "Species residual SDs: separate_v1",
       subtitle = "Per-species sigma; compare width to the single sigma in hierarchical models")

# --- E3. Species-specific parameters: hierarchical_v1 (pooled sex) -----------
# Partial pooling on intercepts and depth slopes; sex is a single global effect.
# Compared to separate_v1, intervals should be narrower (borrowing strength).
# Comparing beta_sex here (a scalar) to hier_v3's vector of beta_sex_s directly
# shows the cost/benefit of assuming sex dimorphism is equal across species.

mcmc_areas(
  subset_draws(draws_h1, variable = c("alpha[1]", "alpha[2]", "alpha[3]")),
  prob = 0.89, point_est = "median"
) +
  scale_y_discrete(labels = species_labs) +
  labs(title    = "Species intercepts: hierarchical_v1 (pooled sex)",
       subtitle = "Partial pooling; compare interval width to separate_v1")

mcmc_areas(
  subset_draws(draws_h1, variable = c("beta_depth[1]", "beta_depth[2]", "beta_depth[3]")),
  prob = 0.89, point_est = "median"
) +
  scale_y_discrete(labels = species_labs) +
  labs(title    = "Species depth slopes: hierarchical_v1 (pooled sex)")

mcmc_areas(
  subset_draws(draws_h1, variable = "beta_sex"),
  prob = 0.89, point_est = "median"
) +
  labs(title    = "Global sex effect: hierarchical_v1",
       subtitle = "Single coefficient — assumed equal across all species")

# --- E4. Species-specific parameters: hierarchical_v3 (varying sex) ----------
# All coefficients vary by species. Compared to hier_v1, the main difference
# is beta_sex becoming a vector. If sigma_sex is estimated close to zero, the
# species-varying sex effect collapses toward the global model (hier_v1).

mcmc_areas(
  subset_draws(draws_h3, variable = c("alpha[1]", "alpha[2]", "alpha[3]")),
  prob = 0.89, point_est = "median"
) +
  scale_y_discrete(labels = species_labs) +
  labs(title    = "Species intercepts: hierarchical_v3 (varying sex)",
       subtitle = "Expected: Gentoo >> Chinstrap > Adelie; tighter than separate_v1")

mcmc_areas(
  subset_draws(draws_h3, variable = c("beta_depth[1]", "beta_depth[2]", "beta_depth[3]")),
  prob = 0.89, point_est = "median"
) +
  scale_y_discrete(labels = species_labs) +
  labs(title    = "Species depth slopes: hierarchical_v3 (varying sex)",
       subtitle = "Separated posteriors justify species-specific slopes")

mcmc_areas(
  subset_draws(draws_h3, variable = c("beta_sex[1]", "beta_sex[2]", "beta_sex[3]")),
  prob = 0.89, point_est = "median"
) +
  scale_y_discrete(labels = species_labs) +
  labs(title    = "Species sex effects: hierarchical_v3 (varying sex)",
       subtitle = "Overlap across species → global sex effect sufficient; separation → v3 needed")

# --- E5. Direct comparison: pooled vs varying sex effect ---------------------
# Side-by-side: hier_v1's single beta_sex vs hier_v3's three beta_sex_s values.
# If the three species-specific intervals from v3 all overlap with each other
# and with v1's pooled estimate, the simpler v1 specification is adequate.
# If one species (likely Gentoo) pulls away, v3's flexibility is warranted.

mcmc_intervals(
  subset_draws(draws_h1, variable = "beta_sex"),
  prob = 0.89
) +
  labs(title    = "hier_v1: single global sex effect",
       subtitle = "Point = median, inner bar = 50% CI, outer = 89% CI")

mcmc_intervals(
  subset_draws(draws_h3, variable = c("beta_sex[1]", "beta_sex[2]", "beta_sex[3]")),
  prob = 0.89
) +
  scale_y_discrete(labels = species_labs) +
  labs(title    = "hier_v3: species-varying sex effects",
       subtitle = "Compare spread to v1's pooled estimate above")

# --- E6. Hyperparameter posteriors: hierarchical models ----------------------
# Unique to the hierarchical models; pooled and separate have no equivalent.
# sigma_alpha vs sigma_beta vs sigma_sex shows which coefficient type varies
# most across species. hier_v3 adds sigma_sex; if it is estimated near zero,
# the pooled sex assumption in hier_v1 is effectively what the data support.

mcmc_areas(
  subset_draws(draws_h1, variable = c("mu_alpha", "mu_beta_depth",
                                      "sigma_alpha", "sigma_beta")),
  prob = 0.89, point_est = "median"
) +
  labs(title    = "Hyperparameters: hierarchical_v1",
       subtitle = "No sigma_sex because sex effect is not hierarchical in v1")

mcmc_areas(
  subset_draws(draws_h3, variable = c("mu_alpha", "mu_beta_depth", "mu_sex",
                                      "sigma_alpha", "sigma_beta", "sigma_sex")),
  prob = 0.89, point_est = "median"
) +
  labs(title    = "Hyperparameters: hierarchical_v3",
       subtitle = "sigma_sex near zero → pooled sex assumption adequate")

# --- E7. MCMC diagnostics — traceplots and convergence summaries -------------
# Hairy caterpillar chains = good mixing. Rhat < 1.01 and ESS_bulk > ~400
# indicate reliable inference. The hierarchical models are the most complex
# and most likely to show sampling issues; check them carefully.

rstan::traceplot(fit_pooled,
                 pars       = c("alpha", "beta_depth", "beta_sex", "sigma"),
                 inc_warmup = FALSE) +
  ggtitle("Traceplots: pooled_v1")

rstan::traceplot(fit_hier_v1,
                 pars       = c("mu_alpha", "mu_beta_depth",
                                "sigma_alpha", "sigma_beta", "beta_sex", "sigma"),
                 inc_warmup = FALSE) +
  ggtitle("Traceplots — hyperparameters: hierarchical_v1")

rstan::traceplot(fit_hier_v1,
                 pars       = c("alpha[1]", "alpha[2]", "alpha[3]",
                                "beta_depth[1]", "beta_depth[2]", "beta_depth[3]"),
                 inc_warmup = FALSE) +
  ggtitle("Traceplots — species-level coefficients: hierarchical_v1")

rstan::traceplot(fit_hier_v3,
                 pars       = c("mu_alpha", "mu_beta_depth", "mu_sex",
                                "sigma_alpha", "sigma_beta", "sigma_sex", "sigma"),
                 inc_warmup = FALSE) +
  ggtitle("Traceplots — hyperparameters: hierarchical_v3")

rstan::traceplot(fit_hier_v3,
                 pars       = c("alpha[1]", "alpha[2]", "alpha[3]",
                                "beta_depth[1]", "beta_depth[2]", "beta_depth[3]",
                                "beta_sex[1]",   "beta_sex[2]",   "beta_sex[3]"),
                 inc_warmup = FALSE) +
  ggtitle("Traceplots — species-level coefficients: hierarchical_v3")

rstan::traceplot(fit_separate,
                 pars       = c("alpha[1]", "alpha[2]", "alpha[3]",
                                "alpha[4]", "alpha[5]", "alpha[6]",
                                "beta[1]",  "beta[2]",  "beta[3]",
                                "beta[4]",  "beta[5]",  "beta[6]",
                                "sigma[1]", "sigma[2]", "sigma[3]",
                                "sigma[4]", "sigma[5]", "sigma[6]"),
                 inc_warmup = FALSE) +
  ggtitle("Traceplots: separate_v1")

cat("\n===== Convergence diagnostics: pooled_v1 =====\n")
print(fit_pooled,
      pars  = c("alpha", "beta_depth", "beta_sex", "beta_species", "sigma"),
      probs = c(0.05, 0.5, 0.95))

cat("\n===== Convergence diagnostics: hierarchical_v1 (pooled sex) =====\n")
print(fit_hier_v1,
      pars  = c("mu_alpha", "mu_beta_depth",
                "sigma_alpha", "sigma_beta", "sigma",
                "alpha", "beta_depth", "beta_sex"),
      probs = c(0.05, 0.5, 0.95))

cat("\n===== Convergence diagnostics: hierarchical_v3 (varying sex) =====\n")
print(fit_hier_v3,
      pars  = c("mu_alpha", "mu_beta_depth", "mu_sex",
                "sigma_alpha", "sigma_beta", "sigma_sex", "sigma",
                "alpha", "beta_depth", "beta_sex"),
      probs = c(0.05, 0.5, 0.95))

cat("\n===== Convergence diagnostics: separate_v1 =====\n")
print(fit_separate,
      pars  = c("alpha", "beta", "sigma"),
      probs = c(0.05, 0.5, 0.95))
# Rhat < 1.01 and ESS_bulk / ESS_tail > ~400 indicate reliable inference.
