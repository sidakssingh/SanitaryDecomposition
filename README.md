# Analyzing Decomposition in Sanitation Systems

Hey there! 👋 Welcome to my science fair project repo. Here, I take a deep dive into the real-world breakdown of hygiene products (like toilet paper and wipes) in sanitation systems—combining hands-on lab experimentation with advanced data science and machine learning. If you’re curious about what really happens to those so-called "flushable" wipes, you’re in the right place!

## Project Overview: Science Meets Data
For this project, I designed and executed a series of controlled, multi-day laboratory trials to rigorously measure the decomposition of various hygiene products. Each product was tracked across multiple time points, with precise mass measurements collected daily. The result? A robust dataset capturing the true breakdown behavior of these products in simulated sanitation environments.

But I didn’t stop at data collection. I built a full analytics pipeline from scratch:
- **Experimental Design:** Developed and ran repeatable lab protocols to ensure reliable, unbiased results.
- **Data Engineering:** Cleaned, transformed, and validated all measurements using Python (pandas, NumPy).
- **Statistical Modeling:** Applied linear regression and repeated measures ANOVA to uncover significant trends and differences between products.
- **Machine Learning:** Trained a neural network (TensorFlow) to predict mass loss over time, demonstrating predictive modeling skills.
- **Scientific Reporting:** Automated the creation of clear, publication-quality plots and summary tables for easy interpretation.
- **Advanced Analysis (R):** Performed post-hoc tests and created interaction plots for deeper insights, using R for complementary statistical exploration.

## Why This Matters
With growing concerns about the environmental impact of hygiene products, especially those labeled as "flushable," this project provides actionable, data-driven insights for consumers, engineers, and policymakers. The approach blends experimental rigor with modern analytics—showcasing both lab and coding skills.

## How I Broke Down the Analysis
- **Python:** This is where all the main action happens—cleaning up the data, running the big stats (like regression and ANOVA), training a neural network, and making the main plots and tables.
- **R:** If you want to go deeper, the R script is for extra analysis—like post-hoc tests and fancier plots. No repeated work here, just bonus insights for the stats nerds (or science fair judges who want more!).

## Getting Started (It’s Easy!)

### 1. What You’ll Need

#### Python
If you haven’t already, install these packages:
```bash
pip install pandas numpy matplotlib statsmodels tensorflow scikit-learn openpyxl
```

#### R
And for R:
```r
install.packages(c('dplyr', 'ggplot2', 'multcomp', 'emmeans'))
```

### 2. Your Data File
Just drop your CSV file in this folder and name it `degradation_data.csv`. It should have columns like:
- `trial_id` (which experiment/trial)
- `product_type` (e.g., Toilet Paper, Wipe, etc.)
- `day` (which day the measurement was taken)
- `mass_g` (mass in grams)

### 3. Run the Code

#### Python (main analysis)
```bash
python analyze_decomposition.py
```

#### R (extra stats & plots)
```r
source('analyze_decomposition.R')
# Or just open it in RStudio and hit Run
```

### 4. What You’ll Get

#### Python Outputs
- `linear_regression_summary.txt`: All the details from the regression model
- `anova_results.txt`: Results from the repeated measures ANOVA
- `nn_predictions.csv`: Neural network predictions vs. actual values
- `decomposition_report.png`: A plot showing how each product breaks down over time
- `decomposition_summary.xlsx`: A handy summary table for quick reference

#### R Outputs (bonus analysis)
- `tukey_posthoc_R.txt`: Which products are really different? This tells you!
- `interaction_plot_R.png`: A colorful plot showing how products compare across days

## Why I Did This
This project started as a science fair entry (and, not to brag, but it won a few awards 🏆). I wanted to see if those “flushable” wipes really live up to the hype—and the results were pretty eye-opening. Along the way, I got to flex my skills in experimental design, data science, and scientific communication. If you’re working on something similar, or just want to geek out over decomposition data, feel free to use this code or reach out with questions.

---

*Thanks for checking out my project! If you have ideas, spot a bug, or want to collaborate, open an issue or pull request. Happy decomposing!* 🌱 