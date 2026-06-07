
library(rstan)
library(loo)
library(posterior)
library(bayesplot)
library(tidyverse)
library(patchwork)

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
species_dummy      <- model.matrix(~ species - 1, data = penguins_clean)[, -1]
group_index        <- mutate(penguins, groups = interaction(species, sex))$groups |> as.integer()

species_labels     <- as.character(penguins_clean$species)
sex_labels         <- as.character(penguins_clean$sex)
species_sex_labels <- paste(penguins_clean$species, penguins_clean$sex, sep = "-")
group_labels       <- mutate(penguins, groups = interaction(species, sex))$groups |> as.character()

y <- bill_length_scaled

# take a species index vector.
stan_data_pooled_t <- list(
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

stan_data_separate_t <- list(
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

REFIT_MODELS <- FALSE
SAVE_FITS    <- TRUE

rds_pooled_t       <- "src/models/fit_pooled_t.RDS"
rds_hierarchical_v3_t <- "src/models/fit_hierarchical_v3_t.RDS"
rds_hierarchical_v1_t <- "src/models/fit_hierarchical_v1_t.RDS"
rds_separate_t     <- "src/models/fit_separate_v3_t.RDS"

if (REFIT_MODELS) {
  fit_pooled_t <- stan(
    file   = "src/models/pooled_v1_t.stan",
    data   = stan_data_pooled_t,
    chains = 4, iter = 2000, warmup = 1000, seed = 187
  )
  fit_hierarchical_v3_t <- stan(
    file   = "src/models/hierarchical_v3_t.stan",
    data   = stan_data_indexed,
    chains = 4, iter = 2000, warmup = 1000, seed = 187
  )
  fit_hierarchical_v1_t <- stan(
    file   = "src/models/hierarchical_v3_t.stan",
    data   = stan_data_indexed,
    chains = 4, iter = 2000, warmup = 1000, seed = 187
  )
  fit_separate_t <- stan(
    file   = "src/models/separate_v3_t.stan",
    data   = stan_data_separate_t,
    chains = 4, iter = 2000, warmup = 1000, seed = 187
  )

  if (SAVE_FITS) {
    saveRDS(fit_pooled_t,       rds_pooled_t)
    saveRDS(fit_hierarchical_v1_t, rds_hierarchical_v1_t)
    saveRDS(fit_hierarchical_v3_t, rds_hierarchical_v3_t)
    saveRDS(fit_separate_t,     rds_separate_t)
    message("Fits saved to src/models/")
  }
} else {
  fit_pooled_t          <- readRDS(rds_pooled_t)
  fit_hierarchical_v3_t <- readRDS(rds_hierarchical_v3_t)
  fit_hierarchical_v1_t <- readRDS(rds_hierarchical_v1_t)
  fit_separate_t        <- readRDS(rds_separate_t)
  message("Fits loaded from src/models/")
}


loo_pooled_t          <- loo(fit_pooled_t)
loo_hierarchical_v3_t <- loo(fit_hierarchical_v3_t)
loo_hierarchical_v1_t <- loo(fit_hierarchical_v1_t)
loo_separate_t        <- loo(fit_separate_t)

# -----------------------------------------------------------------------------
# Individual LOO summaries
# -----------------------------------------------------------------------------


cat("\n===== pooled_t_v1 (complete pooling) =====\n")
print(loo_pooled_t)

cat("\n===== hierarchical_t_v3 (partial pooling, species-specific sex effects) =====\n")
print(loo_hierarchical_v3_t)

cat("\n===== hierarchical_t_v1 (partial pooling, shared sex effect) =====\n")
print(loo_hierarchical_v1_t)

cat("\n===== separate_t_v3 (no pooling) =====\n")
print(loo_separate_t)

# -----------------------------------------------------------------------------
# Head-to-head comparison
# -----------------------------------------------------------------------------

cat("\n===== LOO comparison: all three models =====\n")
comp <- loo_compare(loo_pooled_t, loo_hierarchical_v1_t, loo_hierarchical_v3_t, loo_separate_t)
print(comp)

# -----------------------------------------------------------------------------
# Summary table
# -----------------------------------------------------------------------------

summary_df <- tibble(
  model    = c("Pooled", "Hierarchical (varying sex)", "Hierarchical (pooled sex)", "Separate"),
  elpd_loo = c(
    loo_pooled_t$estimates["elpd_loo", "Estimate"],
    loo_hierarchical_v3_t$estimates["elpd_loo", "Estimate"],
    loo_hierarchical_v1_t$estimates["elpd_loo", "Estimate"],
    loo_separate_t$estimates["elpd_loo", "Estimate"]
  ),
  se_elpd  = c(
    loo_pooled_t$estimates["elpd_loo", "SE"],
    loo_hierarchical_v3_t$estimates["elpd_loo", "SE"],
    loo_hierarchical_v1_t$estimates["elpd_loo", "SE"],
    loo_separate_t$estimates["elpd_loo", "SE"]
  ),
  p_loo    = c(
    loo_pooled_t$estimates["p_loo", "Estimate"],
    loo_hierarchical_v3_t$estimates["p_loo", "Estimate"],
    loo_hierarchical_v1_t$estimates["p_loo", "Estimate"],
    loo_separate_t$estimates["p_loo", "Estimate"]
  ),
  looic    = c(
    loo_pooled_t$estimates["looic", "Estimate"],
    loo_hierarchical_v3_t$estimates["looic", "Estimate"],
    loo_hierarchical_v1_t$estimates["looic", "Estimate"],
    loo_separate_t$estimates["looic", "Estimate"]
  )
) %>% arrange(desc(elpd_loo))

cat("\n===== Summary table (sorted by ELPD, higher = better) =====\n")
print(summary_df, n = Inf)

# -----------------------------------------------------------------------------
# Pareto-k diagnostic plots
# -----------------------------------------------------------------------------

par(mfrow = c(2, 2))
plot(loo_pooled_t,       main = "Pareto-k: Pooled",       label_points = FALSE)
plot(loo_hierarchical_v3_t, main = "Pareto-k: Hierarchical (varying sex)", label_points = FALSE)
plot(loo_hierarchical_v1_t, main = "Pareto-k: Hierarchical (pooled sex)", label_points = FALSE)
plot(loo_separate_t,     main = "Pareto-k: Separate",     label_points = FALSE)
par(mfrow = c(1, 1))

pareto_k_table(loo_pooled_t)
pareto_k_table(loo_hierarchical_v3_t)
pareto_k_table(loo_hierarchical_v1_t)
pareto_k_table(loo_separate_t)

cat("== The one observation id with k > 0.7 ==\n")
pareto_k_ids(loo_separate_t) # 283

# -----------------------------------------------------------------------------
# PPC side-by-side: overall density
# -----------------------------------------------------------------------------

yrep_pooled_t       <- as_draws_matrix(fit_pooled_t)       %>% subset_draws(variable = "yrep")
yrep_hierarchical_t_v3 <- as_draws_matrix(fit_hierarchical_v3_t) %>% subset_draws(variable = "yrep")
yrep_hierarchical_t_v1 <- as_draws_matrix(fit_hierarchical_v1_t) %>% subset_draws(variable = "yrep")
yrep_separate_t     <- as_draws_matrix(fit_separate_t)     %>% subset_draws(variable = "yrep")

p1 <- ppc_dens_overlay(y, yrep_pooled_t) +
  ggtitle("PPC: Pooled")

p3 <- ppc_dens_overlay(y, yrep_hierarchical_t_v3) +
  ggtitle("PPC: Hierarchical (varying sex)")

p2 <- ppc_dens_overlay(y, yrep_hierarchical_t_v1) +
  ggtitle("PPC: Hierarchical (pooled sex)")

p4 <- ppc_dens_overlay(y, yrep_separate_t) +
  ggtitle("PPC: Separate")

wrap_plots(
  p1, p2, p3 ,p4, 
  ncol = 2
)

# -----------------------------------------------------------------------------
# PPC side-by-side: overall min - max
# -----------------------------------------------------------------------------


p1 <- ppc_stat_2d(y, yrep_pooled_t,
            stat = c("min","max")) +
  ggtitle("Min vs Max: Pooled")

p3 <- ppc_stat_2d(y, yrep_hierarchical_t_v3,
            stat = c("min","max")) +
  ggtitle("Min vs Max: Hierarchical (varying sex)")

p2 <- ppc_stat_2d(y, yrep_hierarchical_t_v1,
            stat = c("min","max")) +
  ggtitle("Min vs Max: Hierarchical (pooled sex)")

p4 <- ppc_stat_2d(y, yrep_separate_t,
            stat = c("min","max")) +
  ggtitle("Min vs Max: Separate")

wrap_plots(
  p1,p2,p3,p4, ncol = 2
)
