# Briefing Doc: P Values and Hypothesis Testing

This briefing document summarises key concepts related to p-values and hypothesis testing, drawing primarily from "P Values (Calculated Probability) and Hypothesis Testing - StatsDirect" by Iain E. Buchan.

## Understanding P-values and Hypothesis Testing

The **p-value**, or calculated probability, indicates the likelihood of observing the study's results (or more extreme ones) if the **null hypothesis (H0)** is true. The null hypothesis typically posits "no difference" between groups being compared.

### Key Points:

- **Null Hypothesis (H0):** A statement of no effect or difference. The study aims to find evidence against this hypothesis.
- **Alternative Hypothesis (H1):** The opposite of the null hypothesis, representing the effect or difference the study is investigating.
- **Significance Level (Alpha):** A pre-determined threshold (commonly 0.05, 0.01, or 0.001) used to determine statistical significance.
- **Statistically Significant:** If the p-value is less than the chosen significance level, the null hypothesis is rejected, suggesting the results support the alternative hypothesis.

As stated in the source, 

> "If your P value is less than the chosen significance level then you reject the null hypothesis i.e. accept that your sample gives reasonable evidence to support the alternative hypothesis."

### One-sided vs. Two-sided P-values:

- **Two-sided tests** are generally recommended and consider differences in both directions.
- **One-sided tests** are used only when a large change in one specific direction is relevant to the study.

### Important Considerations:

- **Statistical significance** does not automatically imply practical significance or importance. The real-world relevance of the findings should always be considered.
- While arbitrary, the 5% significance level is widely used: 

> "Conventionally the 5% (less than 1 in 20 chance of being wrong), 1% and 0.1% (P < 0.05, 0.01 and 0.001) levels have been used."

### Types of Error:

- **Type I Error:** Rejecting a true null hypothesis (false positive). The probability of this error is denoted by alpha (the significance level).
- **Type II Error:** Failing to reject a false null hypothesis (false negative). The probability of this error is denoted by beta.

> "The significance level (alpha) is the probability of type I error. The power of a test is one minus the probability of type II error (beta)."

### Power and Confidence Intervals:

- **Power:** The probability of correctly rejecting a false null hypothesis. High power is desirable.
- **Confidence Intervals:** Provide a range within which the true population parameter is likely to lie. They offer valuable information beyond p-values.

The author emphasises that 

> "Statistical referees of scientific journals expect authors to quote confidence intervals with greater prominence than P values."

## Conclusion:

Understanding p-values, hypothesis testing, and associated concepts like Type I/II errors and confidence intervals is crucial for interpreting statistical results in research. While p-values provide a measure of statistical significance, they should not be the sole focus. Researchers must consider the broader context, effect sizes, and practical implications of their findings.
