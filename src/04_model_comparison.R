# =============================================================================
# Model comparison — pooled_v1 vs hierarchical_v3 vs separate_v1
# =============================================================================

library(rstan)
library(loo)
library(posterior)
library(bayesplot)
library(tidyverse)
library(patchwork)

options(mc.cores = parallel::detectCores())
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
species_dummy      <- model.matrix(~ species - 1, data = penguins_clean)[, -1]
group_index        <- mutate(penguins, groups = interaction(species, sex))$groups |> as.integer()

species_labels     <- as.character(penguins_clean$species)
sex_labels         <- as.character(penguins_clean$sex)
species_sex_labels <- paste(penguins_clean$species, penguins_clean$sex, sep = "-")
group_labels       <- mutate(penguins, groups = interaction(species, sex))$groups |> as.character()

y <- bill_length_scaled

# take a species index vector.
stan_data_pooled <- list(
  N           = nrow(penguins_clean),
  bill_length = bill_length_scaled,
  bill_depth  = bill_depth_scaled,
  sex         = sex_dummy,
  species     = species_dummy
)

stan_data_indexed <- list(
  N           = nrow(penguins_clean),
  J           = 3L,
  bill_length = bill_length_scaled,
  bill_depth  = bill_depth_scaled,
  sex         = sex_dummy,
  species     = species_index
)

stan_data_separate <- list(
  N           = nrow(penguins_clean),
  G           = 6L,
  bill_length = bill_length_scaled,
  bill_depth  = bill_depth_scaled,
  group         = group_index
)

# -----------------------------------------------------------------------------
# Fit / load models
# -----------------------------------------------------------------------------
# Set REFIT_MODELS <- TRUE to re-run MCMC from scratch.
# Set REFIT_MODELS <- FALSE to load previously saved .RDS fits from src/models/.
# Set SAVE_FITS <- TRUE to write fits to disk after sampling (ignored when loading).

REFIT_MODELS <- TRUE
SAVE_FITS    <- TRUE

rds_pooled       <- "src/models/fit_pooled_v1.RDS"
rds_hierarchical_v3 <- "src/models/fit_hierarchical_v3.RDS"
rds_hierarchical_v1 <- "src/models/fit_hierarchical_v1.RDS"
rds_separate     <- "src/models/fit_separate_v3.RDS"

if (REFIT_MODELS) {
  fit_pooled <- stan(
    file   = "src/models/pooled_v1.stan",
    data   = stan_data_pooled,
    chains = 4, iter = 2000, warmup = 1000, seed = 187
  )
  fit_hierarchical_v3 <- stan(
    file   = "src/models/hierarchical_v3.stan",
    data   = stan_data_indexed,
    chains = 4, iter = 2000, warmup = 1000, seed = 187
  )
  fit_hierarchical_v1 <- stan(
    file   = "src/models/hierarchical_v1.stan",
    data   = stan_data_indexed,
    chains = 4, iter = 2000, warmup = 1000, seed = 187
  )
  fit_separate <- stan(
    file   = "src/models/separate_v3.stan",
    data   = stan_data_separate,
    chains = 4, iter = 2000, warmup = 1000, seed = 187
  )

  if (SAVE_FITS) {
    saveRDS(fit_pooled,       rds_pooled)
    saveRDS(fit_hierarchical_v3, rds_hierarchical_v3)
    saveRDS(fit_hierarchical_v1, rds_hierarchical_v1)
    saveRDS(fit_separate,     rds_separate)
    message("Fits saved to src/models/")
  }
} else {
  fit_pooled          <- readRDS(rds_pooled)
  fit_hierarchical_v3 <- readRDS(rds_hierarchical_v3)
  fit_hierarchical_v1 <- readRDS(rds_hierarchical_v1)
  fit_separate        <- readRDS(rds_separate)
  message("Fits loaded from src/models/")
}

# -----------------------------------------------------------------------------
# LOO-CV
# -----------------------------------------------------------------------------
loo_pooled          <- loo(fit_pooled)
loo_hierarchical_v1 <- loo(fit_hierarchical_v1)
loo_hierarchical_v3 <- loo(fit_hierarchical_v3)
loo_separate        <- loo(fit_separate)

apply_mm <- function(loo_obj, fit) {
  if (!any(loo_obj$diagnostics$pareto_k > 0.7)) return(loo_obj)
  if (!REFIT_MODELS) {
    warning("High Pareto-k detected but moment_match requires a live compiled ",
            "model. Set REFIT_MODELS <- TRUE and re-run to apply it.")
    return(loo_obj)
  }
  message("High Pareto-k — refitting with moment_match=TRUE")
  loo(fit, moment_match = TRUE)
}

loo_pooled          <- apply_mm(loo_pooled,       fit_pooled)
loo_hierarchical_v1 <- apply_mm(loo_hierarchical_v1, fit_hierarchical_v1)
loo_hierarchical_v3 <- apply_mm(loo_hierarchical_v3, fit_hierarchical_v3)
loo_separate        <- apply_mm(loo_separate,     fit_separate)

# -----------------------------------------------------------------------------
# Individual LOO summaries
# -----------------------------------------------------------------------------

cat("\n===== pooled_v1 (complete pooling) =====\n")
print(loo_pooled)

cat("\n===== hierarchical_v3 (partial pooling, species-specific sex effects) =====\n")
print(loo_hierarchical_v3)

cat("\n===== hierarchical_v1 (partial pooling, shared sex effect) =====\n")
print(loo_hierarchical_v1)

cat("\n===== separate_v3 (no pooling) =====\n")
print(loo_separate)

# -----------------------------------------------------------------------------
# Head-to-head comparison
# -----------------------------------------------------------------------------
cat("\n===== LOO comparison: all three models =====\n")
comp <- loo_compare(loo_pooled, loo_hierarchical_v1, loo_hierarchical_v3, loo_separate)
print(comp)

# -----------------------------------------------------------------------------
# Summary table
# -----------------------------------------------------------------------------

summary_df <- tibble(
  model    = c("Pooled", "Hierarchical (varying sex)", "Hierarchical (pooled sex)", "Separate"),
  elpd_loo = c(
    loo_pooled$estimates["elpd_loo", "Estimate"],
    loo_hierarchical_v3$estimates["elpd_loo", "Estimate"],
    loo_hierarchical_v1$estimates["elpd_loo", "Estimate"],
    loo_separate$estimates["elpd_loo", "Estimate"]
  ),
  se_elpd  = c(
    loo_pooled$estimates["elpd_loo", "SE"],
    loo_hierarchical_v3$estimates["elpd_loo", "SE"],
    loo_hierarchical_v1$estimates["elpd_loo", "SE"],
    loo_separate$estimates["elpd_loo", "SE"]
  ),
  p_loo    = c(
    loo_pooled$estimates["p_loo", "Estimate"],
    loo_hierarchical_v3$estimates["p_loo", "Estimate"],
    loo_hierarchical_v1$estimates["p_loo", "Estimate"],
    loo_separate$estimates["p_loo", "Estimate"]
  ),
  looic    = c(
    loo_pooled$estimates["looic", "Estimate"],
    loo_hierarchical_v3$estimates["looic", "Estimate"],
    loo_hierarchical_v1$estimates["looic", "Estimate"],
    loo_separate$estimates["looic", "Estimate"]
  )
) %>% arrange(desc(elpd_loo))

cat("\n===== Summary table (sorted by ELPD, higher = better) =====\n")
print(summary_df, n = Inf)

# -----------------------------------------------------------------------------
# Pareto-k diagnostic plots
# # -----------------------------------------------------------------------------

par(mfrow = c(2, 2))
plot(loo_pooled,       main = "Pareto-k: Pooled",       label_points = FALSE)
plot(loo_hierarchical_v3, main = "Pareto-k: Hierarchical (varying sex)", label_points = FALSE)
plot(loo_hierarchical_v1, main = "Pareto-k: Hierarchical (pooled sex)", label_points = FALSE)
plot(loo_separate,     main = "Pareto-k: Separate",     label_points = FALSE)
par(mfrow = c(1, 1))

pareto_k_table(loo_pooled)
pareto_k_table(loo_hierarchical_v3)
pareto_k_table(loo_hierarchical_v1)
pareto_k_table(loo_separate)

# -----------------------------------------------------------------------------
# PPC side-by-side: overall density
# -----------------------------------------------------------------------------
yrep_pooled       <- as_draws_matrix(fit_pooled)       %>% subset_draws(variable = "yrep")
yrep_hierarchical_v3 <- as_draws_matrix(fit_hierarchical_v3) %>% subset_draws(variable = "yrep")
yrep_hierarchical_v1 <- as_draws_matrix(fit_hierarchical_v1) %>% subset_draws(variable = "yrep")
yrep_separate     <- as_draws_matrix(fit_separate)     %>% subset_draws(variable = "yrep")

p1 <- ppc_dens_overlay(y, yrep_pooled) +
  ggtitle("PPC: Pooled")

p3 <- ppc_dens_overlay(y, yrep_hierarchical_v3) +
  ggtitle("PPC: Hierarchical (varying sex)")

p2 <- ppc_dens_overlay(y, yrep_hierarchical_v1) +
  ggtitle("PPC: Hierarchical (pooled sex)")

p4 <- ppc_dens_overlay(y, yrep_separate) +
  ggtitle("PPC: Separate")

wrap_plots(
  p1, p2, p3 ,p4, 
  ncol = 2
)

# -----------------------------------------------------------------------------
# PPC side-by-side: min max
# -----------------------------------------------------------------------------

p1 <- ppc_stat_2d(y, yrep_pooled,
            stat = c("min","max")) +
  ggtitle("Min vs Max: Pooled")

p3 <- ppc_stat_2d(y, yrep_hierarchical_v3,
            stat = c("min","max")) +
  ggtitle("Min vs Max: Hierarchical (varying sex)")

p2 <- ppc_stat_2d(y, yrep_hierarchical_v1,
            stat = c("min","max")) +
  ggtitle("Min vs Max: Hierarchical (pooled sex)")

p4 <- ppc_stat_2d(y, yrep_separate,
            stat = c("min","max")) +
  ggtitle("Min vs Max: Separated")

wrap_plots(
  p1,p2,p3,p4, ncol = 2
)
