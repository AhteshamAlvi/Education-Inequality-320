# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
import scipy.stats as stats

df = pd.read_csv("ResearchInformation3.csv")

# All columns are already in the correct datatypes, with strings being objects, 
# and numbers being in float64 (all numbers use decimals so that is appropriate)
# except for the ID, which is an int64
# display(df.dtypes)

#Quick check to see if dataset is clean.
# display(df.isnull().sum())
df['Income'] = df['Income'].str.strip()



# -----------------------------------------------------------------------------------
# Overall Summary Statistics
# ----------------------------------------------------------------------------------
'''
display(df.describe())

states = df.groupby('state')
grade_levels = df.groupby('grade_level')
school_types = df.groupby('school_type')

# Summary Statistics by State
for state,group in states:
    sorted = group.sort_values("school_name", ascending=True)
    display(sorted.describe())
    display(sorted)

# Summary Statistics by Grade Level
for grade,group in grade_levels:
    sorted = group.sort_values("school_name", ascending=True)
    display(sorted.describe())
    display(sorted)

# Summary Statistics by School Type (Public, Private, or Charter)
for school,group in school_types:
    sorted = group.sort_values("school_name", ascending=True)
    display(sorted.describe())
    display(sorted)
'''
# ----------------------------------------------------------------------------------
# Hypothesis
# ----------------------------------------------------------------------------------

# 1) Does Gaming, Job, and Extracurriculurs, affect Preparation?


'''
# Filtering the data to only the relevant columns
x_val = 'percent_minority'
y_val = 'funding_per_student_usd'
df_hyp1 = df[[x_val, y_val]] 

r, p = stats.pearsonr(df_hyp1[x_val], df_hyp1[y_val])

print("Pearson correlation coefficient (r) = ", r)
print("p-value", p)
print("\n")

# Scatter Plot & Regression Line
sns.regplot(x=x_val, y=y_val, data=df_hyp1, line_kws={'color':'red'})
plt.title('Percent Minority vs Funding Per Student (USD)')
plt.xlabel('Percent Minority')
plt.ylabel('Funding Per Student (USD)')
plt.show()
'''

# ----------------------------------------------------------------------------------
# 2) Does the level of income have an impact on the computer proficiency level of a student?
groups = 'Income'
value = 'Computer'

summary = df.groupby(groups)[value].mean()
print(summary)


low = df[df[groups] == 'Low (Below 15,000)'][value]
low_mid  = df[df[groups] == 'Lower middle (15,000-30,000)'][value]
upper_mid = df[df[groups] == 'Upper middle (30,000-50,000)'][value]
high= df[df[groups] == 'High (Above 50,000)'][value]

# Box Plot & ANOVA Test - Post Hoc (Tukey's HSD)

# Boxpolot
sns.boxplot(x=groups, y=value, data=df, palette='deep')
plt.title("Computer Proficiency Distribution by Level of Income")
plt.xlabel("Level of Income")
plt.ylabel("Computer Proficiency (1-5)")
plt.xticks(rotation=30, ha='right')
plt.tight_layout()
plt.show()

# ANOVA
p_value = stats.f_oneway(low, low_mid, upper_mid, high).pvalue
print(f"P-value for one-way ANOVA test: {p_value}")

# Post Hoc (Tukey's HSD)
tukey = stats.tukey_hsd(low, low_mid, upper_mid, high)
print(tukey)

# ----------------------------------------------------------------------------------
# 3) Does the Highschool GPA correlate to College Overall GPA?
x_val = 'HSC'
y_val = 'Overall'
df_hyp3 = df[[x_val, y_val]] 

rho, p = stats.spearmanr(df_hyp3[x_val], df_hyp3[y_val])
print("Spearman's Rank Coefficient (rho) = ", rho)
print("p-value", p)

#Scatter Plot & Regression Analysis & Spearman's Rank Correlation
sns.regplot(x=x_val, y=y_val, data=df_hyp3, line_kws={'color':'red'})
plt.title('Highschool GPA Versus College Overall GPA')
plt.xlabel('Highschool GPA (5.0 Scale)')
plt.ylabel('College Cumulat vGPA (4.0 Scale)')
plt.show()
