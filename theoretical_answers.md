# Part 1: Theoretical Understanding (30%)

## 1. Short Answer Questions

### Q1: Define algorithmic bias and provide two examples of how it manifests in AI systems.

**Algorithmic bias** refers to systematic and unfair discrimination that occurs when AI systems produce results that are systematically prejudiced against certain individuals or groups based on characteristics such as race, gender, age, or socioeconomic status. This bias can arise from biased training data, flawed model design, or unintended correlations in the data.

**Examples:**

1. **Racial Bias in Criminal Justice Systems**: The COMPAS (Correctional Offender Management Profiling for Alternative Sanctions) algorithm used in US courts was found to incorrectly flag black defendants as future criminals at nearly twice the rate as white defendants. This bias manifested through higher false positive rates for black defendants, leading to harsher sentencing recommendations.

2. **Gender Bias in Hiring Algorithms**: Amazon's AI recruiting tool learned to penalize female candidates by associating words like "women's" or "female" with lower scores. The system was trained on historical hiring data that reflected existing gender biases in the tech industry, causing it to perpetuate and amplify these inequalities.

### Q2: Explain the difference between transparency and explainability in AI. Why are both important?

**Transparency** refers to the openness and accessibility of information about how an AI system works, including its architecture, training data, and decision-making processes. It's about making the system's inner workings visible and understandable to stakeholders.

**Explainability** refers to the ability to provide clear, interpretable explanations for individual AI decisions or predictions in human-understandable terms. It focuses on answering the "why" question for specific outputs.

**Why both are important:**

- **Transparency** builds trust by allowing external audits, regulatory compliance, and public understanding of AI systems. It enables stakeholders to assess fairness, identify biases, and ensure accountability.

- **Explainability** is crucial for practical use cases where users need to understand specific decisions (e.g., why a loan was denied, why a medical diagnosis was made). It supports human oversight and enables users to contest or appeal decisions.

- Together, they create a comprehensive framework for responsible AI deployment that balances technical capability with human understanding and oversight.

### Q3: How does GDPR (General Data Protection Regulation) impact AI development in the EU?

GDPR significantly impacts AI development in the EU through several key provisions:

1. **Right to Explanation (Article 22)**: Individuals have the right to meaningful information about automated decision-making processes, including profiling. This requires AI systems to provide explanations for decisions that significantly affect individuals.

2. **Data Minimization and Purpose Limitation**: AI systems must only collect and process data that is necessary for specified, explicit, and legitimate purposes. This limits the scope of data collection for AI training.

3. **Consent Requirements**: Clear, informed consent is required for data processing, making it challenging to use personal data for AI training without explicit permission.

4. **Right to Erasure**: Individuals can request deletion of their personal data, which may require AI models to be retrained or updated to remove specific individuals' influence.

5. **Privacy by Design**: AI systems must incorporate data protection measures from the initial design stage, requiring privacy-preserving techniques like federated learning or differential privacy.

6. **Data Protection Impact Assessments**: High-risk AI applications require formal assessments of potential privacy risks and mitigation strategies.

These requirements push EU AI developers toward more privacy-preserving, explainable, and ethically designed systems.

## 2. Ethical Principles Matching

**A) Justice** → Fair distribution of AI benefits and risks.

**B) Non-maleficence** → Ensuring AI does not harm individuals or society.

**C) Autonomy** → Respecting users' right to control their data and decisions.

**D) Sustainability** → Designing AI to be environmentally friendly.

---

*This section demonstrates fundamental understanding of AI ethics principles and their practical applications in real-world AI development scenarios.*
