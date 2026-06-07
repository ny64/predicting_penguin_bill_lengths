library(ggplot2)
library(dplyr)

penguins <- readRDS("data/penguins.RDS")

sum(is.na(penguins))
# [1] 0

sapply(penguins, class)
#    species        sex   bill_depth  bill_length 
#   "factor"   "factor"    "numeric"    "numeric" 

head(penguins)
#   species    sex bill_depth bill_length
# 1  Adelie   male       18.7        39.1
# 2  Adelie female       17.4        39.5
# 3  Adelie female       18.0        40.3
# 5  Adelie female       19.3        36.7
# 6  Adelie   male       20.6        39.3
# 7  Adelie female       17.8        38.9

tail(penguins)
#       species    sex bill_depth bill_length
# 339 Chinstrap female       17.0        45.7
# 340 Chinstrap   male       19.8        55.8
# 341 Chinstrap female       18.1        43.5
# 342 Chinstrap   male       18.2        49.6
# 343 Chinstrap   male       19.0        50.8
# 344 Chinstrap female       18.7        50.2

summary(penguins)
#       species        sex        bill_depth     bill_length   
#  Adelie   :146   female:165   Min.   :13.10   Min.   :32.10  
#  Chinstrap: 68   male  :168   1st Qu.:15.60   1st Qu.:39.50  
#  Gentoo   :119                Median :17.30   Median :44.50  
#                               Mean   :17.16   Mean   :43.99  
#                               3rd Qu.:18.70   3rd Qu.:48.60  
#                               Max.   :21.50   Max.   :59.60  

penguins |> group_by(species, sex) |> summarise(n = n(), mu = mean(bill_length), std = sd(bill_length), rho = cor(bill_depth, bill_length))
# A tibble: 6 × 6
# Groups:   species [3]
#   species   sex        n    mu   std     rho
#   <fct>     <fct>  <int> <dbl> <dbl>   <dbl>
# 1 Adelie    female    73  37.3  2.03  0.161 
# 2 Adelie    male      73  40.4  2.28 -0.0382
# 3 Chinstrap female    34  46.6  3.11  0.256 
# 4 Chinstrap male      34  51.1  1.56  0.446 
# 5 Gentoo    female    58  45.6  2.05  0.430 
# 6 Gentoo    male      61  49.5  2.72  0.307 

species_colours <- c(
  Adelie    = "#4472C4",  # blue
  Chinstrap = "#00eeff",  # orange
  Gentoo    = "#ff7700"   # green
)
set_theme(theme_bw())
update_theme(legend.position = "top")

# == Variable distributions ====================================================
# questionable if all of this is actually usefull ? cant see anything in these histograms imho
barplot(table(penguins$species))
barplot(table(penguins$sex))
hist(penguins$bill_length)   # outcome
hist(penguins$bill_depth)    # predictor

# Bill length (outcome) and bill depth (predictor) by sex

hist(penguins$bill_length[penguins$sex == "male"],
     col  = rgb(0.2, 0.4, 0.8, 0.5),
     main = "Bill Length by Sex",
     xlab = "mm",
     xlim = range(penguins$bill_length, na.rm = TRUE),
     ylim = c(0, 35),
     breaks = 15)
hist(penguins$bill_length[penguins$sex == "female"],
     col    = rgb(0.8, 0.2, 0.2, 0.5),
     breaks = 15,
     add    = TRUE)
legend("topright", legend = c("Male", "Female"),
       fill = c(rgb(0.2, 0.4, 0.8, 0.5), rgb(0.8, 0.2, 0.2, 0.5)))

hist(penguins$bill_depth[penguins$sex == "male"],
     col  = rgb(0.2, 0.4, 0.8, 0.5),
     main = "Bill Depth by Sex",
     xlab = "mm",
     xlim = range(penguins$bill_depth, na.rm = TRUE),
     ylim = c(0, 30),
     breaks = 15)
hist(penguins$bill_depth[penguins$sex == "female"],
     col    = rgb(0.8, 0.2, 0.2, 0.5),
     breaks = 15,
     add    = TRUE)
legend("topright", legend = c("Male", "Female"),
       fill = c(rgb(0.2, 0.4, 0.8, 0.5), rgb(0.8, 0.2, 0.2, 0.5)))

# Bill length by species (the outcome variable — species have very different
# baseline bill lengths, motivating a species-level intercept)

adelie    <- penguins$bill_length[penguins$species == "Adelie"]
gentoo    <- penguins$bill_length[penguins$species == "Gentoo"]
chinstrap <- penguins$bill_length[penguins$species == "Chinstrap"]

hist(adelie,
     col    = adjustcolor(species_colours["Adelie"],    alpha.f = 0.6),
     main   = "Bill Length by Species",
     xlab   = "Bill Length (mm)",
     ylab   = "Frequency",
     xlim   = range(penguins$bill_length, na.rm = TRUE),
     ylim   = c(0, 40),
     breaks = 15)
hist(gentoo,
     col    = adjustcolor(species_colours["Gentoo"],    alpha.f = 0.6),
     breaks = 15,
     add    = TRUE)
hist(chinstrap,
     col    = adjustcolor(species_colours["Chinstrap"], alpha.f = 0.6),
     breaks = 15,
     add    = TRUE)
legend("topright",
       legend = names(species_colours),
       fill   = adjustcolor(species_colours, alpha.f = 0.6),
       border = "black",
       title  = "Species")

# == Scatter plots: bill_depth vs bill_length ==================================
#
# bill_length is the outcome (y-axis); bill_depth is the predictor (x-axis).
# These plots are the main motivation for the hierarchical model.

# By sex only
ggplot(penguins, aes(x = bill_depth, y = bill_length, colour = sex)) +
  geom_point(alpha = 0.6) +
  geom_smooth(method = "lm", se = FALSE) +
  theme_minimal() +
  labs(title = "Bill length vs. depth by sex",
       x = "Bill depth (mm)", y = "Bill length (mm)")

# By species 
ggplot(penguins, aes(x = bill_depth, y = bill_length, colour = species)) +
  geom_point(alpha = 0.6) +
  geom_smooth(method = "lm", se = FALSE) +
  scale_colour_manual(values = species_colours) +
  theme_minimal() +
  labs(title = "Bill length vs. depth by species (Simpson's paradox)",
       x = "Bill depth (mm)", y = "Bill length (mm)")

# By species AND sex
ggplot(penguins, aes(x = bill_depth, y = bill_length,
                     colour = species, linetype = sex)) +
  geom_point(aes(shape = sex), alpha = 0.6) +
  geom_smooth(method = "lm", se = FALSE) +
  scale_colour_manual(values = species_colours) +
  theme_minimal() +
  labs(title = "Bill length vs. depth by species and sex",
       x = "Bill depth (mm)", y = "Bill length (mm)")

# == Grand box-plot (showing all covariates and target) ========================

summary_data <- penguins %>%
  group_by(species, sex) %>%
  summarize(
    mean_x = mean(bill_depth),
    sd_x = sd(bill_depth),
    
    mean_y = mean(bill_length),
    sd_y = sd(bill_length),
    .groups = "drop"
  )

ggplot() +
  geom_errorbarh(data = summary_data, 
                 aes(x = mean_x, y = mean_y, xmin = mean_x - sd_x, xmax = mean_x + sd_x, color = sex),
                 height = 0.5, linewidth = .6) +
  geom_errorbar(data = summary_data, 
                aes(x = mean_x, y = mean_y, ymin = mean_y - sd_y, ymax = mean_y + sd_y, color = sex),
                width = 0.5, linewidth = .6) +
  
  geom_point(data = summary_data, 
             aes(x = mean_x, y = mean_y, color = sex), 
             size = 4, shape = 18) + 
  
  facet_wrap(~species, nrow = 1) +
  scale_color_manual(values = c("female" = "#ff5722", "male" = "#1a237e")) +
  theme_bw() +
  labs(
    x = "Bill Depth (mm)",
    y = "Bill Length (mm)",
    color = "Sex"
  )


# == some consice plot alternatives ============================================

bill_length_boxplot <- ggplot(data = penguins) +
  geom_boxplot(aes(x = species, y = bill_length, color = species)) +
  scale_color_manual(values = species_colours) +
  facet_wrap(~sex) +
  labs(title = "Boxplot of Bill length by Species and Sex",
       y = "Bill length (mm)")
bill_length_boxplot
# ggsave("figures/bill_length_boxplot.png")
# easy outliers, location, scale and skweness

bill_length_hist <- ggplot(data = penguins, aes(x = bill_length)) +
  geom_histogram(
    aes(y = after_stat(density), fill = sex),
    color = "white",
    position = "identity" ) +
  geom_density(position = "stack") +
  facet_grid(cols = vars(species), rows= vars(sex)) +
  labs(title = "Histogram-Density of Bill length by Species and Sex",
       x = "Bill length (mm)", y = "Bill length (mm)")
bill_length_hist
# ggsave("figures/bill_length_hist.png")
# rough idea of distribution for each group