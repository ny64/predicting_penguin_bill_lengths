# =============================================================================
# Report figures and tables (may not be exhaustive)
# =============================================================================

library(rstan)
library(loo)
library(bayesplot)
library(posterior)
library(tidyverse)
library(patchwork)

options(mc.cores = parallelly::availableCores())
rstan_options(auto_write = TRUE)

dir.create("out/figures", recursive = TRUE, showWarnings = FALSE)

# colour palette — one colour per model, consistent across all plots
MODEL_COLS <- c(
  pooled        = "#4E79A7",
  hier_v1       = "#F28E2B",
  hier_v3       = "#59A14F",
  separate      = "#E15759"
)
MODEL_LABELS <- c(
  pooled   = "Pooled",
  hier_v1  = "Hierarchical v1\n(pooled sex)",
  hier_v3  = "Hierarchical v3\n(varying sex)",
  separate = "Separate v3\n(6 groups)"
)

color_scheme_set("blue")

# -----------------------------------------------------------------------------
# Data
# -----------------------------------------------------------------------------

penguins <- readRDS("data/penguins.RDS")

penguins_clean <- penguins |>
  drop_na(bill_length, bill_depth, sex, species)

bill_length_scaled <- scale(penguins_clean$bill_length)[, 1]
bill_depth_scaled  <- scale(penguins_clean$bill_depth)[, 1]
sex_dummy          <- as.integer(penguins_clean$sex == "male")
species_index      <- as.integer(penguins_clean$species)
species_dummy      <- model.matrix(~ species - 1, data = penguins_clean)[, -1]
group_index        <- interaction(penguins_clean$species, penguins_clean$sex) |>
                        as.integer()

y                  <- bill_length_scaled
species_labels     <- as.character(penguins_clean$species)
sex_labels         <- as.character(penguins_clean$sex)
sp_sex_labels      <- paste(penguins_clean$species, penguins_clean$sex, sep = "\n")

# Group label order matches interaction() levels (alphabetical species within sex)
GROUP_LABS <- c(
  "[1]" = "Adelie\nfemale",    "[2]" = "Chinstrap\nfemale",
  "[3]" = "Gentoo\nfemale",    "[4]" = "Adelie\nmale",
  "[5]" = "Chinstrap\nmale",   "[6]" = "Gentoo\nmale"
)
SPECIES_LABS <- c("[1]" = "Adelie", "[2]" = "Chinstrap", "[3]" = "Gentoo")

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
  group       = group_index
)

# -----------------------------------------------------------------------------
# Load / refit models
# -----------------------------------------------------------------------------

REFIT_MODELS <- FALSE
SAVE_FITS    <- TRUE

rds_pooled   <- "src/models/fit_pooled_v1.RDS"
rds_hier_v1  <- "src/models/fit_hierarchical_v1.RDS"
rds_hier_v3  <- "src/models/fit_hierarchical_v3.RDS"
rds_separate <- "src/models/fit_separate_v3.RDS"

if (REFIT_MODELS) {
  fit_pooled   <- stan("src/models/pooled_v1.stan",       data = stan_data_pooled,
                       chains = 4, iter = 2000, warmup = 1000, seed = 187)
  fit_hier_v1  <- stan("src/models/hierarchical_v1.stan", data = stan_data_indexed,
                       chains = 4, iter = 2000, warmup = 1000, seed = 187)
  fit_hier_v3  <- stan("src/models/hierarchical_v3.stan", data = stan_data_indexed,
                       chains = 4, iter = 2000, warmup = 1000, seed = 187)
  fit_separate <- stan("src/models/separate_v3.stan",     data = stan_data_separate,
                       chains = 4, iter = 2000, warmup = 1000, seed = 187)
  if (SAVE_FITS) {
    saveRDS(fit_pooled,   rds_pooled)
    saveRDS(fit_hier_v1,  rds_hier_v1)
    saveRDS(fit_hier_v3,  rds_hier_v3)
    saveRDS(fit_separate, rds_separate)
  }
} else {
  fit_pooled   <- readRDS(rds_pooled)
  fit_hier_v1  <- readRDS(rds_hier_v1)
  fit_hier_v3  <- readRDS(rds_hier_v3)
  fit_separate <- readRDS(rds_separate)
}

draws_p  <- as_draws_matrix(fit_pooled)
draws_h1 <- as_draws_matrix(fit_hier_v1)
draws_h3 <- as_draws_matrix(fit_hier_v3)
draws_s  <- as_draws_matrix(fit_separate)

yrep_pooled   <- subset_draws(draws_p,  variable = "yrep")
yrep_hier_v1  <- subset_draws(draws_h1, variable = "yrep")
yrep_hier_v3  <- subset_draws(draws_h3, variable = "yrep")
yrep_separate <- subset_draws(draws_s,  variable = "yrep")

# =============================================================================
# TABLE 1 — Convergence diagnostics
# =============================================================================
# Rhat and ESS for key parameters in each model. Printed to console; copy
# into the report appendix. Flag any Rhat > 1.01 or ESS < 400.

conv_summary <- function(fit, pars, label) {
  s <- summary(fit, pars = pars, probs = c(0.05, 0.5, 0.95))$summary
  as_tibble(s, rownames = "parameter") |>
    select(parameter, mean, sd, `5%`, `50%`, `95%`, n_eff, Rhat) |>
    mutate(model = label, .before = 1)
}

cat("\n===== TABLE 1: Convergence diagnostics =====\n")

conv_pooled <- conv_summary(
  fit_pooled,
  c("alpha", "beta_depth", "beta_sex", "beta_species[1]", "beta_species[2]", "sigma"),
  "Pooled"
)

conv_hier_v1 <- conv_summary(
  fit_hier_v1,
  c("mu_alpha", "mu_beta_depth", "sigma_alpha", "sigma_beta",
    "alpha[1]", "alpha[2]", "alpha[3]",
    "beta_depth[1]", "beta_depth[2]", "beta_depth[3]",
    "beta_sex", "sigma"),
  "Hierarchical v1"
)

conv_hier_v3 <- conv_summary(
  fit_hier_v3,
  c("mu_alpha", "mu_beta_depth", "mu_sex",
    "sigma_alpha", "sigma_beta", "sigma_sex",
    "alpha[1]", "alpha[2]", "alpha[3]",
    "beta_depth[1]", "beta_depth[2]", "beta_depth[3]",
    "beta_sex[1]", "beta_sex[2]", "beta_sex[3]", "sigma"),
  "Hierarchical v3"
)

conv_separate <- conv_summary(
  fit_separate,
  c("alpha[1]", "alpha[2]", "alpha[3]", "alpha[4]", "alpha[5]", "alpha[6]",
    "beta[1]",  "beta[2]",  "beta[3]",  "beta[4]",  "beta[5]",  "beta[6]",
    "sigma[1]", "sigma[2]", "sigma[3]", "sigma[4]", "sigma[5]", "sigma[6]"),
  "Separate v3"
)

conv_all <- bind_rows(conv_pooled, conv_hier_v1, conv_hier_v3, conv_separate) |>
  mutate(across(where(is.numeric), \(x) round(x, 3)))
print(conv_all, n = Inf)

# Flag any problems
bad <- filter(conv_all, Rhat > 1.01 | n_eff < 400)
if (nrow(bad) > 0) {
  cat("\nWARNING — parameters with Rhat > 1.01 or ESS < 400:\n")
  print(bad)
} else {
  cat("\nAll Rhat < 1.01 and ESS > 400. Sampling looks healthy.\n")
}

# =============================================================================
# TABLE 2 — LOO-CV comparison
# =============================================================================

loo_pooled   <- loo(fit_pooled)
loo_hier_v1  <- loo(fit_hier_v1)
loo_hier_v3  <- loo(fit_hier_v3)
loo_separate <- loo(fit_separate)

apply_mm <- function(loo_obj, fit, label) {
  if (!any(loo_obj$diagnostics$pareto_k > 0.7)) return(loo_obj)
  if (!REFIT_MODELS) {
    warning(label, ": high Pareto-k but moment_match needs a live model.",
            " Set REFIT_MODELS <- TRUE to apply it.")
    return(loo_obj)
  }
  message(label, ": refitting with moment_match=TRUE")
  loo(fit, moment_match = TRUE)
}

loo_pooled   <- apply_mm(loo_pooled,   fit_pooled,   "Pooled")
loo_hier_v1  <- apply_mm(loo_hier_v1,  fit_hier_v1,  "Hier v1")
loo_hier_v3  <- apply_mm(loo_hier_v3,  fit_hier_v3,  "Hier v3")
loo_separate <- apply_mm(loo_separate, fit_separate, "Separate")

cat("\n===== TABLE 2: LOO-CV comparison =====\n")
comp <- loo_compare(loo_pooled, loo_hier_v1, loo_hier_v3, loo_separate)
print(comp, digits = 1)

loo_tbl <- tibble(
  Model        = c("Pooled", "Hierarchical v1", "Hierarchical v3", "Separate v3"),
  ELPD         = c(loo_pooled$estimates["elpd_loo", "Estimate"],
                   loo_hier_v1$estimates["elpd_loo", "Estimate"],
                   loo_hier_v3$estimates["elpd_loo", "Estimate"],
                   loo_separate$estimates["elpd_loo", "Estimate"]),
  SE           = c(loo_pooled$estimates["elpd_loo", "SE"],
                   loo_hier_v1$estimates["elpd_loo", "SE"],
                   loo_hier_v3$estimates["elpd_loo", "SE"],
                   loo_separate$estimates["elpd_loo", "SE"]),
  p_loo        = c(loo_pooled$estimates["p_loo", "Estimate"],
                   loo_hier_v1$estimates["p_loo", "Estimate"],
                   loo_hier_v3$estimates["p_loo", "Estimate"],
                   loo_separate$estimates["p_loo", "Estimate"])
) |>
  arrange(desc(ELPD)) |>
  mutate(
    delta_elpd = ELPD - max(ELPD),
    across(where(is.numeric), \(x) round(x, 1))
  )

cat("\nFormatted LOO table (sorted best to worst):\n")
print(loo_tbl, n = Inf)

# =============================================================================
# FIGURE 1 — Density overlay: global fit (all four models, 2×2 panel)
# =============================================================================

p_dens <- function(yrep, title) {
  ppc_dens_overlay(y, yrep[1:100, ]) +
    ggtitle(title) +
    theme(legend.position = "none",
          plot.title = element_text(size = 10))
}

fig1 <- (
  p_dens(yrep_pooled,   "Pooled") +
  p_dens(yrep_hier_v1,  "Hierarchical v1 (pooled sex)") +
  p_dens(yrep_hier_v3,  "Hierarchical v3 (varying sex)") +
  p_dens(yrep_separate, "Separate v3 (6 groups)")
) +
  plot_annotation(
    title   = "Figure 1 — Posterior predictive density overlays",
    subtitle = "Thick line: observed. Thin lines: 100 posterior predictive draws.",
    theme   = theme(plot.title    = element_text(size = 12, face = "bold"),
                    plot.subtitle = element_text(size = 9))
  ) +
  plot_layout(ncol = 2)

ggsave("out/figures/fig1_density_overlay.pdf", fig1,
       width = 10, height = 7)
print(fig1)

# =============================================================================
# FIGURE 2 — Species×sex cell means (the key discriminating check)
# =============================================================================
# This is the sharpest PPC: 6 cells, each with ~30–80 obs.
# Separate v3 encodes sex structurally (one group per species×sex cell),
# so it should nail these means. Hier v1's single beta_sex may struggle
# if the male-female gap genuinely differs by species.

p_spsx <- function(yrep, title) {
  ppc_stat_grouped(y, yrep, group = sp_sex_labels, stat = "mean") +
    ggtitle(title) +
    theme(legend.position = "none",
          plot.title       = element_text(size = 10),
          axis.text.y      = element_text(size = 7))
}

fig2 <- (
  p_spsx(yrep_pooled,   "Pooled") +
  p_spsx(yrep_hier_v1,  "Hierarchical v1") +
  p_spsx(yrep_hier_v3,  "Hierarchical v3") +
  p_spsx(yrep_separate, "Separate v3")
) +
  plot_annotation(
    title    = "Figure 2 — Posterior predictive species × sex cell means",
    subtitle = "Histogram: T(y_rep). Vertical line: T(y). All four models should cover the line.",
    theme    = theme(plot.title    = element_text(size = 12, face = "bold"),
                     plot.subtitle = element_text(size = 9))
  ) +
  plot_layout(ncol = 2)

ggsave("out/figures/fig2_spxsex_means.pdf", fig2,
       width = 10, height = 9)
print(fig2)

# =============================================================================
# FIGURE 3 — Sex effect: pooled (hier v1) vs varying (hier v3) vs separate
# =============================================================================
# Direct comparison of how each model represents the sex dimorphism.
# Hier v1: one scalar beta_sex
# Hier v3: three species-specific beta_sex[s] from a shared hyperprior
# Separate v3: sex is implicit — compare alpha[male] - alpha[female] per species

# Derived sex contrast for separate_v3
# Groups: 1=Adelie.f, 2=Chins.f, 3=Gent.f, 4=Adelie.m, 5=Chins.m, 6=Gent.m
sex_contrast_sep <- bind_rows(
  tibble(
    species  = "Adelie",
    contrast = as.numeric(draws_s[, "alpha[4]"]) - as.numeric(draws_s[, "alpha[1]"])
  ),
  tibble(
    species  = "Chinstrap",
    contrast = as.numeric(draws_s[, "alpha[5]"]) - as.numeric(draws_s[, "alpha[2]"])
  ),
  tibble(
    species  = "Gentoo",
    contrast = as.numeric(draws_s[, "alpha[6]"]) - as.numeric(draws_s[, "alpha[3]"])
  )
)

# Hier v1: replicate the single beta_sex for all three species for plotting
sex_contrast_h1 <- tibble(
  species  = rep(c("Adelie", "Chinstrap", "Gentoo"), each = nrow(draws_h1)),
  contrast = rep(as.numeric(draws_h1[, "beta_sex"]), 3)
)

sex_contrast_h3 <- bind_rows(
  tibble(species = "Adelie",    contrast = as.numeric(draws_h3[, "beta_sex[1]"])),
  tibble(species = "Chinstrap", contrast = as.numeric(draws_h3[, "beta_sex[2]"])),
  tibble(species = "Gentoo",    contrast = as.numeric(draws_h3[, "beta_sex[3]"]))
)

sex_contrast_all <- bind_rows(
  mutate(sex_contrast_h1,  model = "Hier v1\n(pooled sex)"),
  mutate(sex_contrast_h3,  model = "Hier v3\n(varying sex)"),
  mutate(sex_contrast_sep, model = "Separate v3\n(alpha contrast)")
) |>
  mutate(
    model   = factor(model,   levels = c("Hier v1\n(pooled sex)",
                                         "Hier v3\n(varying sex)",
                                         "Separate v3\n(alpha contrast)")),
    species = factor(species, levels = c("Adelie", "Chinstrap", "Gentoo"))
  )

fig3 <- ggplot(sex_contrast_all, aes(x = contrast, fill = model, colour = model)) +
  geom_density(alpha = 0.35, linewidth = 0.6) +
  facet_wrap(~ species, ncol = 1, scales = "free_y") +
  geom_vline(xintercept = 0, linetype = "dashed", colour = "grey40") +
  scale_fill_manual(values   = c("#F28E2B", "#59A14F", "#E15759")) +
  scale_colour_manual(values = c("#F28E2B", "#59A14F", "#E15759")) +
  labs(
    title    = "Figure 3 — Sex effect by species across models",
    subtitle = "Hier v1: single global beta_sex (replicated per species for comparison).\nHier v3: species-specific beta_sex from hyperprior. Separate: alpha[male] − alpha[female].",
    x        = "Sex effect (male − female, scaled bill length)",
    y        = "Density",
    fill     = NULL, colour = NULL
  ) +
  theme_default(base_family = "sans") +
  theme(
    legend.position = "bottom",
    strip.text      = element_text(face = "bold")
  )

ggsave("out/figures/fig3_sex_effect.pdf", fig3,
       width = 7, height = 8)
print(fig3)

# =============================================================================
# FIGURE 4 — Separate v3: per-group posteriors (the selected model)
# =============================================================================
# Full parameter inspection for the selected model.

fig4a <- mcmc_areas(
  subset_draws(draws_s, variable = paste0("alpha[", 1:6, "]")),
  prob = 0.89, point_est = "median"
) +
  scale_y_discrete(labels = GROUP_LABS) +
  labs(title    = "Intercepts — separate v3",
       subtitle = "One per species × sex group")

fig4b <- mcmc_areas(
  subset_draws(draws_s, variable = paste0("beta[", 1:6, "]")),
  prob = 0.89, point_est = "median"
) +
  scale_y_discrete(labels = GROUP_LABS) +
  labs(title    = "Depth slopes — separate v3")

fig4c <- mcmc_areas(
  subset_draws(draws_s, variable = paste0("sigma[", 1:6, "]")),
  prob = 0.89, point_est = "median"
) +
  scale_y_discrete(labels = GROUP_LABS) +
  labs(title    = "Residual SDs — separate v3")

fig4 <- (fig4a / fig4b / fig4c) +
  plot_annotation(
    title = "Figure 4 — Posterior marginals: separate v3 (selected model)",
    theme = theme(plot.title = element_text(size = 12, face = "bold"))
  )

ggsave("out/figures/fig4_separate_posteriors.pdf", fig4,
       width = 7, height = 12)
print(fig4)

# =============================================================================
# FIGURE 5 — Traceplots: separate v3 (convergence visual)
# =============================================================================

fig5_alpha <- rstan::traceplot(
  fit_separate,
  pars       = paste0("alpha[", 1:6, "]"),
  inc_warmup = FALSE
) + ggtitle("Traceplots: intercepts — separate v3")

fig5_beta <- rstan::traceplot(
  fit_separate,
  pars       = paste0("beta[", 1:6, "]"),
  inc_warmup = FALSE
) + ggtitle("Traceplots: depth slopes — separate v3")

fig5_sigma <- rstan::traceplot(
  fit_separate,
  pars       = paste0("sigma[", 1:6, "]"),
  inc_warmup = FALSE
) + ggtitle("Traceplots: residual SDs — separate v3")

pdf("out/figures/fig5_traceplots_separate.pdf", width = 10, height = 10)
print(fig5_alpha)
print(fig5_beta)
print(fig5_sigma)
dev.off()

print(fig5_alpha)
print(fig5_beta)
print(fig5_sigma)

# =============================================================================
# FIGURE 6 — LOO ELPD comparison bar chart
# =============================================================================

loo_plot_df <- loo_tbl |>
  mutate(
    Model = factor(Model, levels = rev(c("Pooled", "Hierarchical v1",
                                         "Hierarchical v3", "Separate v3")))
  )

fig6 <- ggplot(loo_plot_df, aes(x = ELPD, y = Model)) +
  geom_point(size = 3, colour = "#333333") +
  geom_errorbarh(aes(xmin = ELPD - SE, xmax = ELPD + SE),
                 height = 0.25, colour = "#333333") +
  geom_vline(xintercept = max(loo_plot_df$ELPD),
             linetype = "dashed", colour = "grey50") +
  labs(
    title    = "Figure 6 — LOO-CV expected log predictive density",
    subtitle = "Higher ELPD = better out-of-sample predictive accuracy. Error bars = ±1 SE.",
    x        = "ELPD (LOO)",
    y        = NULL
  ) +
  theme_default(base_family = "sans")

ggsave("out/figures/fig6_loo_comparison.pdf", fig6,
       width = 7, height = 4)
print(fig6)

# =============================================================================
# FIGURE 7 — Pareto-k diagnostics
# =============================================================================

pdf("out/figures/fig7_pareto_k.pdf", width = 10, height = 4)
par(mfrow = c(1, 4))
plot(loo_pooled,   main = "Pareto-k: Pooled",        label_points = FALSE)
plot(loo_hier_v1,  main = "Pareto-k: Hier v1",        label_points = FALSE)
plot(loo_hier_v3,  main = "Pareto-k: Hier v3",        label_points = FALSE)
plot(loo_separate, main = "Pareto-k: Separate v3",    label_points = FALSE)
par(mfrow = c(1, 1))
dev.off()

par(mfrow = c(1, 4))
plot(loo_pooled,   main = "Pareto-k: Pooled",        label_points = FALSE)
plot(loo_hier_v1,  main = "Pareto-k: Hier v1",        label_points = FALSE)
plot(loo_hier_v3,  main = "Pareto-k: Hier v3",        label_points = FALSE)
plot(loo_separate, main = "Pareto-k: Separate v3",    label_points = FALSE)
par(mfrow = c(1, 1))

# =============================================================================
# FIGURE 8 — Species-grouped density: all four models (report supplement)
# =============================================================================

p_spdens <- function(yrep, title) {
  ppc_dens_overlay_grouped(y, yrep[1:100, ], group = species_labels) +
    ggtitle(title) +
    theme(legend.position = "none",
          plot.title = element_text(size = 9),
          strip.text = element_text(size = 8))
}

fig8 <- (
  p_spdens(yrep_pooled,   "Pooled") +
  p_spdens(yrep_hier_v1,  "Hierarchical v1") +
  p_spdens(yrep_hier_v3,  "Hierarchical v3") +
  p_spdens(yrep_separate, "Separate v3")
) +
  plot_annotation(
    title    = "Figure 8 — Per-species posterior predictive density overlays",
    subtitle = "Thick: observed. Thin: 100 draws.",
    theme    = theme(plot.title    = element_text(size = 12, face = "bold"),
                     plot.subtitle = element_text(size = 9))
  ) +
  plot_layout(ncol = 2)

ggsave("out/figures/fig8_species_density.pdf", fig8,
       width = 10, height = 8)
print(fig8)

cat("\n===== All figures saved to out/figures/ =====\n")
cat("fig1_density_overlay.pdf     — global density check (all 4 models)\n")
cat("fig2_spxsex_means.pdf        — species×sex cell means (key PPC)\n")
cat("fig3_sex_effect.pdf          — sex effect comparison across models\n")
cat("fig4_separate_posteriors.pdf — posterior marginals: selected model\n")
cat("fig5_traceplots_separate.pdf — traceplots: selected model\n")
cat("fig6_loo_comparison.pdf      — LOO ELPD bar chart\n")
cat("fig7_pareto_k.pdf            — Pareto-k diagnostics\n")
cat("fig8_species_density.pdf     — per-species density (supplement)\n")
