# Analyzing Decomposition in Sanitation Systems (R Pipeline)
#
# Welcome! This R script is for extra, deeper-dive analysis that goes beyond the basics handled in Python.
# Here you'll find things like post-hoc tests and more nuanced plots—stuff that's great for science fair judges or anyone who loves stats.
# (No repeated work from Python—promise!)
#
# Required packages: dplyr, ggplot2, multcomp, emmeans
# If you haven't already, run: install.packages(c('dplyr', 'ggplot2', 'multcomp', 'emmeans'))

library(dplyr)
library(ggplot2)
library(multcomp)
library(emmeans)

# --- Load the same CSV as Python ---
csv_file <- 'degradation_data.csv'
df <- read.csv(csv_file, stringsAsFactors = FALSE)

# --- Make sure percent_remaining is there (just in case) ---
# If the Python script already made it, great! If not, we'll whip it up here.
if (!'percent_remaining' %in% colnames(df)) {
  if (!('mass_g' %in% colnames(df) && 'trial_id' %in% colnames(df))) {
    stop('Your CSV needs either percent_remaining or both mass_g and trial_id columns.')
  }
  df <- df %>% group_by(trial_id) %>% mutate(initial_mass = first(mass_g)) %>% ungroup()
  df <- df %>% mutate(percent_remaining = mass_g / initial_mass * 100)
}

# --- Post-hoc Tukey HSD: which products are REALLY different? ---
# After the main ANOVA in Python, this digs into which pairs of products differ, day by day.
lm_model <- lm(percent_remaining ~ day * product_type, data = df)
emm <- emmeans(lm_model, ~ product_type | day)
posthoc <- contrast(emm, method = "pairwise", adjust = "tukey")
capture.output(summary(posthoc), file = 'tukey_posthoc_R.txt')

# --- Interaction plot: see how products behave over time ---
# This plot shows the average percent remaining for each product, each day. It's a great way to spot trends or weird outliers.
p <- ggplot(df, aes(x = day, y = percent_remaining, color = product_type, group = product_type)) +
  stat_summary(fun = mean, geom = "line", size = 1) +
  stat_summary(fun = mean, geom = "point", size = 2) +
  labs(title = 'Interaction Plot: Decomposition by Product Type and Day',
       x = 'Day', y = '% Mass Remaining') +
  theme_minimal()
ggsave('interaction_plot_R.png', plot = p, width = 8, height = 5)

cat('Done with the extra stats! Check tukey_posthoc_R.txt for details on which products differ, and interaction_plot_R.png for a cool visual summary.\n') 