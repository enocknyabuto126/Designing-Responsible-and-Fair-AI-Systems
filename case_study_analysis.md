# Part 2: Case Study Analysis (40%)

## Case 1: Amazon's Biased Hiring Tool

### Scenario Summary
Amazon developed an AI recruiting tool that used machine learning to evaluate job candidates. The system was trained on historical hiring data from the company over a 10-year period, which predominantly consisted of male candidates. The AI learned to penalize female candidates by associating words like "women's," "female," and references to women's colleges with lower scores.

### Source of Bias Analysis

**Primary Source: Training Data Bias**
The fundamental issue was that the training data reflected historical gender imbalances in Amazon's tech workforce. The AI system learned patterns from data where:
- Male candidates were overrepresented in successful applications
- Female candidates were underrepresented, especially in technical roles
- The system learned to associate male-dominated language patterns with higher scores

**Secondary Sources:**
- **Feature Engineering**: The system may have used features that indirectly encoded gender information
- **Model Design**: Lack of explicit fairness constraints or bias mitigation techniques
- **Feedback Loops**: The system would have reinforced existing biases by continuing to favor male candidates

### Three Proposed Fixes

#### Fix 1: Data Augmentation and Balancing
- **Implementation**: Collect additional training data from underrepresented groups and use techniques like SMOTE (Synthetic Minority Over-sampling Technique) to balance the dataset
- **Benefits**: Addresses the root cause by providing more diverse training examples
- **Challenges**: Requires significant effort to collect new data and may not capture all relevant patterns

#### Fix 2: Fairness-Aware Model Training
- **Implementation**: Integrate fairness constraints using techniques like adversarial debiasing, reweighting, or demographic parity constraints
- **Benefits**: Actively prevents the model from learning biased patterns during training
- **Challenges**: May slightly reduce overall accuracy while improving fairness

#### Fix 3: Post-Processing Bias Correction
- **Implementation**: Apply post-processing techniques like equalized odds postprocessing or calibration to adjust model outputs
- **Benefits**: Can be applied to existing models without retraining
- **Challenges**: May not address underlying bias in the model's learned representations

### Fairness Evaluation Metrics

#### Primary Metrics:
1. **Disparate Impact Ratio**: Ratio of positive outcomes between protected groups (should be close to 1.0)
2. **Equal Opportunity Difference**: Difference in true positive rates between groups (should be close to 0)
3. **Statistical Parity Difference**: Difference in positive prediction rates between groups

#### Secondary Metrics:
4. **False Positive Rate Difference**: Ensures similar error rates across groups
5. **Calibration**: Ensures predicted probabilities are well-calibrated for all groups
6. **Individual Fairness**: Measures consistency of predictions for similar individuals

## Case 2: Facial Recognition in Policing

### Ethical Risks Analysis

#### 1. Wrongful Arrests and Convictions
- **Risk Level**: HIGH
- **Impact**: False positive identifications can lead to innocent people being detained, arrested, or convicted
- **Example**: In 2020, Robert Williams was wrongfully arrested in Detroit due to facial recognition misidentification, spending 30 hours in jail

#### 2. Privacy Violations and Mass Surveillance
- **Risk Level**: HIGH
- **Impact**: Continuous monitoring of public spaces without consent violates reasonable expectations of privacy
- **Concerns**: Chilling effect on free speech and assembly, potential for government overreach

#### 3. Racial and Gender Bias Amplification
- **Risk Level**: HIGH
- **Impact**: Higher error rates for minorities and women can lead to disproportionate targeting and harassment
- **Evidence**: Studies show error rates up to 10x higher for darker-skinned individuals

#### 4. Due Process Violations
- **Risk Level**: MEDIUM
- **Impact**: Reliance on "black box" technology without proper validation or human oversight
- **Concerns**: Lack of transparency in decision-making processes

#### 5. Function Creep and Mission Drift
- **Risk Level**: MEDIUM
- **Impact**: Technology designed for one purpose being used for broader surveillance
- **Examples**: Using facial recognition for tracking protesters, monitoring public spaces

### Responsible Deployment Policies

#### 1. Strict Use Case Limitations
- **Policy**: Limit facial recognition to specific, high-value criminal investigations only
- **Implementation**: Require judicial warrants for each use case
- **Prohibitions**: Ban use in public spaces, protests, or general surveillance

#### 2. Accuracy and Bias Requirements
- **Policy**: Mandate minimum accuracy thresholds (e.g., 99.5% for positive identifications)
- **Implementation**: Regular bias audits and performance monitoring
- **Requirements**: Equal error rates across demographic groups

#### 3. Human Oversight and Validation
- **Policy**: Require human verification of all AI-generated matches
- **Implementation**: Multiple officer review process for identifications
- **Training**: Mandatory training on AI limitations and bias recognition

#### 4. Transparency and Accountability
- **Policy**: Public reporting on system usage, accuracy, and bias metrics
- **Implementation**: Regular public audits and independent oversight
- **Requirements**: Clear documentation of all decisions and their rationale

#### 5. Right to Contest and Appeal
- **Policy**: Establish clear procedures for individuals to contest identifications
- **Implementation**: Independent review boards and appeal processes
- **Protections**: Right to know when facial recognition was used in their case

#### 6. Sunset Provisions and Regular Review
- **Policy**: Require regular reauthorization of facial recognition programs
- **Implementation**: Annual performance reviews and public hearings
- **Requirements**: Evidence of effectiveness and fairness for continued use

---

*These case studies demonstrate the complex ethical challenges in AI deployment and the importance of proactive bias identification and mitigation strategies.*
