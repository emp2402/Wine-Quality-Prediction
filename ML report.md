#  **Wine Quality Prediction via Supervised Machine Learning Classification**

| Pouya Sotoudeh1 | Federica Brasca1 | Emmanuele Pucci1 | Takudzwa Rioga1 |
| :---- | :---- | :---- | :---- |
| Student ID: 946920 | Student ID: 907082 | Student ID: 939157 | Student ID: 946902 |

1  *Università degli Studi di Milano-Bicocca, CdLM Data Science*  
---

**Abstract**

Predicting wine quality from physicochemical measurements is inherently difficult due to subjective sensory labels, overlapping feature distributions across adjacent quality tiers, and pronounced class imbalance at the extremes. This study proposes a cost-sensitive framework for ordinal wine-quality assessment, implemented as a fully reproducible KNIME workflow and applied to red Vinho Verde wines.

Rather than optimizing models solely for accuracy, evaluation is guided by an asymmetric cost structure that reflects the unequal impact of different misclassification types, particularly severe errors involving high-quality wines. Several classifiers are compared under stratified 10-fold cross-validation, revealing that a tuned Gradient Boosting model offers the most effective balance between predictive performance and cost sensitivity. The results illustrate how incorporating domain-informed error costs can lead to more meaningful model selection in early-stage, data-driven wine quality estimation.

**Keywords:** Asymmetric Cost Matrix, Class Imbalance, Classification, Cost-Sensitive Evaluation, KNIME, Machine Learning, Physicochemical Analysis, Vinho Verde, Wine Quality.

**Table of Contents**

[1\. Introduction	1](#1.-introduction)

[1.1 Research Questions	1](#1.1-research-questions)

[1.2 Dataset	2](#heading=h.xij65vitusu4)

[2\. Data Preparation and Processing	2](#2.-data-preparation-and-processing)

[2.1 EDA	2](#2.1-eda)

[2.2 Data Transformation	2](#2.2-data-transformation)

[2.3 Data Partitioning and Normalization	3](#2.3-data-partitioning-and-normalization)

[3\. Classification Models & Methodology	3](#3.-classification-models-&-methodology)

3.1 Cost-Sensitive Model Selection Criterion	3

[3.2 Training and Validation Protocol	3](#3.2-training-and-validation-protocol)

[3.3 Evaluated Classification Models	4](#3.3-evaluated-classification-models)

[4\. Model Evaluation	4](#4.-model-evaluation)

[4.1 Cost Matrix Design	4](#4.1-cost-matrix-design)

[4.2 Performance Metrics and Findings	4](#4.2-performance-metrics-and-findings)

[4.3 Hyperparameter Tuning](#4.3-hyperparameter-tuning-–-gradient-boosting)	5

[4.4 Model Comparison and Trade-offs	6](#4.4-model-comparison-and-trade-offs)

[4.5 Feature Importance	7](#4.5-feature-importance)

[5\. Conclusion	8](#5.-conclusion)

[5.1 Limitations and Improvements	8](#5.1-limitations-and-improvements)

[References	8](#references)

## **1\. Introduction** {#1.-introduction}

Wine quality is traditionally assessed primarily through sensory evaluation rather than purely chemical analysis \[1\], making quality grading costly and time-consuming. We investigate whether machine learning can predict subjective quality tiers using only physicochemical measurements, so routine lab tests could support early-stage quality estimation and reduce reliance on manual tasting.

### **1.1 Research Questions** {#1.1-research-questions}

This study addresses two main questions regarding the link between chemical composition and sensory evaluation.

***RQ1:** Is it possible to accurately classify a wine’s quality tier by relying solely on objective physicochemical measurements?*

***RQ2:** Is it possible to identify the primary chemical attributes that are most predictive of a wine’s quality?*

   
**1.2 Dataset**  
The study utilizes the Wine Quality Dataset, sourced from the Kaggle platform \[2\] and comprising 1,143 records of red Portuguese Vinho Verde wines.  
The target variable, quality, is an ordinal sensory score (0–10), while 11 physicochemical numerical attributes serve as independent predictors. The identifier attribute (id) was discarded as it carries no predictive information.

The features utilized in this study describe the wine's chemical profile:

- ***Acidity and pH:*** fixed acidity, volatile acidity, citric acid, and pH, all of which influence freshness, balance, and microbial stability.

- ***Stability-related additives:*** free and total sulfur dioxide (SO₂) and sulphates, which are used to inhibit oxidation and microbial growth, thereby affecting shelf life and aroma preservation.

- ***Body & Composition***: residual sugar, chlorides (salt content), density, and alcohol content, contributing to mouthfeel, structure, and overall sensory perception.

## **2\. Data Preparation and Processing** {#2.-data-preparation-and-processing}

All data preparation and analysis steps were implemented in KNIME, following a reproducible workflow from ingestion to model-ready data.

### 

### **2.1 EDA** {#2.1-eda}

After importing the dataset with the CSV Reader node, we performed basic data integrity checks. No missing values were found, and no duplicate rows were detected. However, further statistical analysis revealed three critical characteristics that dictated our subsequent preprocessing strategy: 

***2.1.1. Target Distribution and Imbalance***  
The original target variable (quality, 0–10) is ordinal and highly imbalanced. In practice, most wines fall in the mid-range, while extreme scores are rare, as shown in Figure 1\. In this dataset, no instances have scores below 3 or above 8\. This imbalance can bias models toward the majority classes and makes standard accuracy less informative for minority classes.

![][image1]  
      **Figure 1\.** Distribution of Wine Quality scores

***2.1.2 Feature Correlations and Drivers***  
To explore relationships between predictors and the target, we examined Pearson correlation coefficients generated by the Linear Correlation node.   
The clearest associations were observed for alcohol (*r \= \+0.48*), the strongest positive correlate, indicating an association whereby wines with higher alcohol content tend to receive higher quality ratings, and volatile acidity (*r \= −0.41*), a strong negative correlate, suggesting that elevated acetic acid, often linked to vinegar-like off-notes, is associated with lower perceived quality.

***2.1.3. Distribution and Outliers***  
While most physicochemical features exhibit approximately bell-shaped distributions, a statistical analysis performed through the Statistics node highlighted that chlorides and residual sugar exhibit high positive skewness (respectively *skew \= 6.03* and *skew \= 4.4*).  
This indicates the presence of outliers, wines with unusually high salt or sugar content, which could significantly distort the learning process and degrade the performance of predictive models if not properly handled.

### 

### **2.2 Data Transformation**  {#2.2-data-transformation}

***2.2.1. Feature Selection***  
The id attribute, identified as a unique identifier with no predictive power, was removed using the Column Filter node. This step was essential to prevent the models from learning spurious correlations based on a unique identifier that carries no information about wine quality, ensuring that predictions are based solely on the physicochemical properties of the wine.

***2.2.2. Quality Discretization***  
The ordinal target variable, quality, shows under-representation of extreme classes. To reduce sparsity and create a more stable classification task, we discretized the original quality score into four ordered tiers using the Rule Engine node:

- ***Poor:*** quality \= 3, 4  
- ***Below Average:*** quality \= 5  
- ***Above Average:*** quality \= 6  
- ***Excellent:*** quality \= 7, 8

This discretization represents a deliberate trade-off between granularity and robustness. Grouping the rare extreme scores (3–4 and 7–8) into broader categories increases sample support, enabling the model to learn clear signals of low and high quality. Moreover, the dominant middle classes were not merged into a single “Average” category. By separating scores 5 and 6, the model is forced to learn subtle physicochemical differences between mediocre and decent wines, a particularly relevant distinction in mass-market wine production.  
The resulting class distribution is shown in Figure 2\. While the middle tiers remain dominant, reclassifying the target into four tiers reduces skewness and provides increased sample support at the extremes, allowing models to identify characteristic physicochemical patterns associated with poor and excellent wines without being overwhelmed by central classes.

![][image2]  
**Figure 2\.**  Absolute Frequencies of the Discretized Quality Tiers shown in the following order: *Above Average, Below Average, Excellent, Poor*

### 

### **2.3 Data Partitioning and Normalization** {#2.3-data-partitioning-and-normalization}

To ensure a reliable and robust evaluation of predictive performance, we adopted a stratified 10-fold Cross-Validation strategy, which is particularly well-suited for datasets of limited size and with imbalanced class distributions.

The dataset was divided into ten mutually exclusive folds of approximately equal size, preserving the original distribution of the four quality tiers within each fold. In each iteration, one fold served as the test set while the remaining nine formed the training set, ensuring that each observation was used for evaluation exactly once and for training in the remaining folds. Preserving class proportions across folds reduced variance in performance estimates and ensured that all models were consistently evaluated on minority categories.

Next, feature scaling was applied to the physicochemical predictors to account for their differing ranges (e.g., total sulfur dioxide exceeding 200 versus density varying by less than 0.01), which can negatively affect distance-based algorithms and gradient descent convergence. To prevent information leakage, normalization was performed strictly within the cross-validation loop. We utilized Z-score normalization via the Normalizer node to rescale the physicochemical predictors to a mean of 0 and a standard deviation of 1\. By calculating the mean and standard deviation only on the training folds and subsequently applying them to the test fold, we ensured that the evaluation remained truly representative of the models' ability to generalize to unseen data.

The use of a 10-fold approach represents a balance between statistical reliability and computational efficiency, providing a robust foundation for the model evaluation and tuning procedures described in the following sections.

## **3\. Classification Models & Methodology** {#3.-classification-models-&-methodology}

Having refined the dataset and established a robust partitioning protocol, the study transitions to the modeling phase. The objective is to evaluate how different algorithmic approaches capture the physicochemical patterns of wine quality.

Rather than treating this as a generic classification task, the modeling process is governed by the inherent ordinal nature of the quality tiers (*Poor*, *Below Average*, *Above Average*, *Excellent*). Accordingly, before introducing the individual classifiers, we define the criterion used to compare and select models.

**3.1 Cost-Sensitive Model Selection Criterion**  
Standard classification metrics (e.g., accuracy) assume all errors have equal importance, an assumption that is unrealistic in wine quality assessment. In a real-world production context, the impact of a prediction error is proportional to its severity: misclassifying an Excellent wine as Poor (or vice versa) entails far greater economic and reputational risk than confusing adjacent quality tiers. Consequently, model selection was guided by a dual-objective framework that balances overall predictive accuracy with the minimization of "critical" misclassifications. 

To address this, we adopted a custom asymmetric cost matrix that assigns higher penalties to more severe misclassifications, increasing with the distance between true and predicted classes. The matrix was manually defined based on domain considerations rather than generic literature defaults.

The cost matrix was applied during evaluation to compute a Total Misclassification Cost. In KNIME, this was implemented using a Rule Engine node to map each actual-predicted class pair to its corresponding cost, followed by aggregation across predictions. This approach ensured that models were evaluated not only on how frequently they made errors, but also on how costly those errors were from a business perspective. As a result, the final model is not only mathematically accurate, but also practically reliable for domain application.

### **3.2 Training and Validation Protocol** {#3.2-training-and-validation-protocol}

All models were trained and evaluated under a unified execution protocol, consistently applying the stratified 10-fold cross-validation strategy introduced in Section 2.3. This iterative process ensured that every observation in the dataset was used exactly once for testing, yielding a complete set of out-of-sample predictions.

Within this framework, the Normalization node was dynamically fitted on the nine training folds of each iteration. This approach strictly isolated the test fold from the scaling parameters, maintaining the integrity of the unseen data simulation.

### **3.3 Evaluated Classification Models** {#3.3-evaluated-classification-models}

To compare how model structure and complexity impact performance under cost-sensitive evaluation, we evaluated diverse supervised algorithms ranging from simple instance-based learners to complex ensemble and neural models.

- ***Random Forest***

Random Forest is an ensemble of decision trees trained on bootstrap samples with randomized feature selection at each split. This design reduces variance and captures non-linear feature interactions with limited hyperparameter tuning. Due to its robustness on structured data, Random Forest served as a reliable, accuracy-oriented baseline.

- ***Multi-Layer Perceptron***

A Multi-Layer Perceptron was used to model complex non-linear relationships between physicochemical features and wine quality. Its architecture was intentionally kept shallow to mitigate overfitting on the moderately sized dataset.

- ***Gradient Boosting***

Gradient Boosting Trees (GBTs) were employed as a sequential ensemble method, where each learner iteratively corrects the residual errors of its predecessor. With appropriate regularization, GBTs are able to capture complex non-linear patterns while maintaining strong generalization performance.

- ***Support Vector Machine (SVM)***

By employing non-linear kernel transformations, the SVM can capture complex relationships that may elude simpler models. This geometric approach focuses on maximizing decision margins between classes, contributing to structural stability and good generalization performance.

- ***K-Nearest Neighbors***

KNN served as a non-parametric baseline, classifying instances based on local neighborhood structures. While intuitive, its sensitivity to feature scaling and class imbalance relegates it to a lower-bound benchmark for more complex models.

## **4\. Model Evaluation** {#4.-model-evaluation}

After training each model, we evaluated their performance using both traditional accuracy and the custom cost metric. 

### **4.1 Cost Matrix Design** {#4.1-cost-matrix-design}

The highest penalty (150) was assigned to underestimating quality, specifically predicting Poor when the wine is Excellent. In a production context, failing to identify a premium product leads to significant lost revenue. This error was therefore weighted more heavily than overestimation, predicting Poor wines as Excellent, which carries a lower, though still substantial, penalty of 100 due to reputational risk. The cost matrix is designed so that penalties increase proportionally with the distance between the true and predicted quality tiers. For errors involving adjacent categories, smaller penalties (range 4 to 10\) are assigned, as these represent minor sensory discrepancies that are less likely to trigger significant business consequences.

This asymmetric weighting encourages the model selection process to favor classifiers that are conservative with respect to high-quality wines, prioritizing solutions that preserve the distinction of Excellent products. The full asymmetric cost structure is presented in Table 1\.

| Actual \\ Predicted | Poor | Below Avg | Above Avg | Excellent |
| :---- | :---: | :---: | :---: | :---: |
| **Poor** | 0 | 5 | 40 | 100 |
| **Below Avg** | 8 | 0 | 4 | 30 |
| **Above Avg** | 25 | 6 | 0 | 7 |
| **Excellent** | 150 | 50 | 10 | 0 |

                               **Table 1\.** Asymmetric Cost Matrix

### **4.2 Performance Metrics and Findings** {#4.2-performance-metrics-and-findings}

To rigorously assess model viability, we employed a dual-metric evaluation strategy. While predictive accuracy serves as a statistical baseline for general reliability, total misclassification cost acts as a proxy for error severity. This framework was not used to optimize a single metric in isolation, but to expose the operational behavior of the classifiers. By contrasting these two measures, we distinguish between models that minimize cost through conservative majority-class predictions and those that incur higher costs due to increased sensitivity to critical minority classes. The evaluation produced the results summarized in Table 2\.

- **Baseline Models (KNN, SVM, MLP)**: The simpler algorithms struggled to capture the physicochemical complexity of wine quality. KNN (accuracy \= 51.3%) and SVM (accuracy \= 56.7%) served as lower-bound baselines, yielding the highest total costs due to frequent confusion between distant quality tiers. The MLP provided a moderate improvement (accuracy \= 61.2%) by modeling non-linear relationships, but failed to generalize sufficiently to match the performance of tree-based ensembles.  
    
- **Ensemble Methods (Random Forest & Gradient Boosting)**: Tree-based ensembles dominated the evaluation. Random Forest achieved the highest overall accuracy (67.6%) and the lowest total cost (2,932), largely by concentrating predictions in the majority classes and avoiding extreme quality assignments such as the Poor tier (Table 3). In contrast, Gradient Boosting achieved a comparable accuracy (67.5%) but incurred a higher total cost (3,225). However, it demonstrated greater sensitivity to minority classes, particularly the Poor quality tier, as shown in Table 4\.

| Model | Accuracy | Cost |
| :---- | :---: | :---: |
| **Random Forest** | 67.6% | 2932 |
| **Gradient Boosting** | 67.5% | 3225 |
| **MLP** | 61.2% | 4091 |
| **SVM** | 56.7% | 4453 |
| **KNN** | 51.3% | 5964 |

**Table 2\.** Summary of Model Results

Although Random Forest exhibited superior aggregate performance in terms of accuracy and total cost, it proved ineffective at identifying the Poor quality class, yielding a recall of zero for this category (Table 3). Due to this inability to generalize to minority instances, Gradient Boosting was selected as the final model. As discussed in the following section, subsequent hyperparameter optimization further improved its performance under both accuracy and cost-based criteria.

|  | Predicted Poor | Predicted Below Avg | Predicted Above Avg | Predicted Excellent |
| :---: | :---: | :---: | :---: | :---: |
| **Actual Poor** | 0 | 30 | 9 | 0 |
| **Actual Below Avg** | 2 | 369 | 104 | 8 |
| **Actual Above Avg** | 1 | 116 | 310 | 35 |
| **Actual Excellent** | 0 | 4 | 59 | 96 |

**Table 3\.** Random Forest Confusion Matrix

|  | Predicted Poor | Predicted Below Avg | Predicted Above Avg | Predicted Excellent |
| :---: | :---: | :---: | :---: | :---: |
| **Actual Poor** | 3 | 24 | 12 | 0 |
| **Actual Below Avg** | 6 | 358 | 113 | 6 |
| **Actual Above Avg** | 4 | 110 | 313 | 35 |
| **Actual Excellent** | 0 | 8 | 54 | 97 |

             **Table 4\.** Gradient Boosting Confusion Matrix

### **4.3 Hyperparameter Tuning  –  Gradient Boosting** {#4.3-hyperparameter-tuning-–-gradient-boosting}

During model development, we conducted an in-depth tuning of the Gradient Boosting classifier, as its performance was sensitive to the choice of tree depth,  number of trees and learning rate. 

#### **4.3.1 Learning Rate**

In Gradient Boosting, each new tree is built to correct the residual errors made by the previous ensemble. The learning rate controls the contribution of each new tree to the overall model. We tested three values:

- **Low Learning Rate (0.05):** Updates are more conservative, typically requiring more trees to reach optimal performance. This setting often improves generalization at the cost of increased computation.  
- **Moderate Learning Rate (0.1):** Balances learning speed and stability.  
- **Higher Learning Rate (0.2):** Increases the influence of individual trees, which accelerates convergence but raises the risk of fitting noise in the training data.

#### **4.3.2 Depth versus Trees**

We tested tree depths of 2, 3, and 4 in combination with 50, 100, and 200 boosting iterations.

- **Depth 4**: At the maximum depth considered in our experiments, the model was able to capture more complex decision boundaries. For example, at a learning rate of 0.05, increasing the number of trees from 50 to 100 improved accuracy from 64.7% to 67.2% while reducing total cost from 3,331 to 3,187. Expanding the ensemble to 200 trees yielded smaller additional gains, with accuracy reaching 68% and cost marginally decreasing to 3,133 (Table 6). These results indicate diminishing returns beyond 100 trees. A similar pattern is observed at a learning rate of 0.1 (Table 5), where increasing the ensemble from 100 to 200 trees leads to reduced accuracy and higher cost, suggesting limited additional generalization benefit.  
- **Depth 3**: At a learning rate of 0.05, a depth-3 model achieved 65.5% accuracy with 200 trees. A configuration with 100 trees yielded one of the lowest costs (3,276) but at the expense of lower accuracy (63.9%). This suggests that while simpler models may reduce severe misclassifications, they fail to capture enough structure to achieve strong overall accuracy.  
- **Depth 2**: Strong underfitting was observed for depth-2 models, with a maximum accuracy of 62.9% at 200 trees and learning rate 0.05. Increasing the learning rate to 0.2 slightly improved accuracy but resulted in higher total costs (Table 7). Costs in this regime remained higher than those of the depth-4 configurations, likely because overly simple models misclassify not only extreme cases but also moderately difficult instances.

| Total Cost | Accuracy | Depth vs  \# Of Trees | Learning Rate |
| :---: | :---: | :---: | :---: |
| 3334 | 65.3% | 4-50 | 0.1 |
| 3225 | 67.5% | 4-100 | 0.1 |
| 3330 | 66.9% | 4-200 | 0.1 |
| 3225 | 64.5% | 3-50 | 0.1 |
| 3404 | 65.2% | 3-100 | 0.1 |
| 3503 | 66.0% | 3-200 | 0.1 |
| 3425 | 61.3% | 2-50 | 0.1 |
| 3284 | 63.3% | 2-100 | 0.1 |
| 3495 | 62.7% | 2-200 | 0.1 |

      **Table 5\.** Comparison of Total Cost to Depth versus Number of Trees for Learning Rate of 0.1

| Total Cost | Accuracy | Depth vs  \# Of Trees | Learning Rate |
| :---: | :---: | :---: | :---: |
| 3331 | 64.7% | 4-50 | 0.05 |
| 3187 | 67.2% | 4-100 | 0.05 |
| 3133 | 68.0% | 4-200 | 0.05 |
| 3521 | 62.2% | 3-50 | 0.05 |
| 3276 | 63.9% | 3-100 | 0.05 |
| 3312 | 65.5% | 3-200 | 0.05 |
| 3782 | 61.1% | 2-50 | 0.05 |
| 3461 | 61.5% | 2-100 | 0.05 |
| 3349 | 62.9% | 2-200 | 0.05 |

**Table 6\.** Comparison of Total Cost to Depth versus Number of Trees for Learning Rate of 0.05

| Total Cost | Accuracy | Depth vs  \# Of Trees | Learning Rate |
| :---: | :---: | :---: | :---: |
| 3263 | 67.1% | 4-50 | 0.2 |
| 3239 | 67.6% | 4-100 | 0.2 |
| 3267 | 66.3% | 4-200 | 0.2 |
| 3496 | 64.7% | 3-50 | 0.2 |
| 3336 | 65.8% | 3-100 | 0.2 |
| 3528 | 65.9% | 3-200 | 0.2 |
| 3441 | 62.3% | 2-50 | 0.2 |
| 3437 | 63.0% | 2-100 | 0.2 |
| 3530 | 63.7% | 2-200 | 0.2 |

**Table 7\.** Comparison of Total Cost to Depth versus Number of Trees for Learning Rate of 0.2

#### **4.3.3 Hyperparameter Tuning Results**

Based on these experiments, a configuration with maximum depth 4, 200 trees, and a learning rate of 0.05 was selected as the final Gradient Boosting model. This setting achieved the highest cross-validated accuracy while maintaining a competitive total misclassification cost (Table 6). Shallower models lacked sufficient representational capacity, while deeper configurations were not explored to avoid unnecessary complexity and to prevent the onset of overfitting. Depth 4 therefore appeared sufficient to capture the complexity of the dataset.

### **4.4 Model Comparison and Trade-offs** {#4.4-model-comparison-and-trade-offs}

Comparing the models reveals clear trade-offs between raw predictive accuracy and cost-sensitive performance. The simpler classifiers (KNN and SVM) were not competitive, indicating that the physicochemical relationships underlying wine quality require more expressive models. The neural network (MLP) showed moderate success; however, its accuracy of 61.2% remained well below that of the tree-based ensembles, and the corresponding reduction in total cost was insufficient to close the gap. This suggests that the additional model complexity introduced by the MLP did not translate into superior performance on the available data.

Between the two ensemble methods, Random Forest achieved the lowest total misclassification cost by concentrating predictions in majority classes and avoiding extreme quality assignments. However, this strategy came at the expense of minority-class detection: despite its favorable aggregate performance, the model failed to identify any instances of the *Poor* class (Recall \= 0). As a necessary operational constraint, any candidate model exhibiting zero recall for the Poor quality tier was deemed unacceptable, regardless of its overall accuracy or total misclassification cost.

Under this constraint, Gradient Boosting, after tuning, delivered comparable accuracy while maintaining non-zero sensitivity to minority quality tiers. Although its total cost was slightly higher, it provided a more balanced trade-off between cost minimization and class sensitivity. Given the study’s objective of reliably identifying all quality tiers, even those with limited representation in the data, Gradient Boosting is therefore a more appropriate model.

### **4.5 Feature Importance** {#4.5-feature-importance}

To assess the contribution of individual features to model predictions, we employed Permutation Feature Importance, using Cohen’s Kappa as the evaluation metric. Alternative approaches, such as a Generalized Linear Model (GLM) or surrogate models, were explored but ultimately discarded. The GLM produced unstable coefficient estimates, while surrogate Random Forest models, trained to approximate the predictions of the Gradient Boosting classifier, introduce an additional layer of model approximation that conflicts with the goal of transparent interpretation.

Permutation Feature Importance directly evaluates the trained model by randomly shuffling one feature at a time and measuring the resulting decrease in Cohen’s Kappa. This approach avoids the instability of linear coefficients and provides a model-agnostic estimate of each feature’s contribution to predictive performance.

Cohen’s Kappa was chosen over cost as the performance metric for feature importance because it provides a stable, class‑imbalance‑robust proxy for overall agreement, whereas cost is strongly influenced by a small number of high‑penalty events. To reduce variance from random shuffling, feature permutation was repeated 20 times and the results were averaged. This procedure ensures that the reported importance values reflect consistent contributions to model performance rather than noise from individual permutations.

## **![][image3]**

**Figure 3\.**  Chart of Attribute ΔKappa Scores for Excellent Wine

| Attribute | ΔKappa |
| :---: | :---: |
| Alcohol | 0.271 |
| Sulphates | 0.158 |
| Volatile Acidity | 0.113 |
| Chlorides | 0.092 |
| Fixed Acidity  | 0.047 |
| Citric Acid | 0.047 |
| Free Sulphur Dioxide | 0.038 |

**Table 8\.** Permutation Importance excluding attributes with ΔKappa \< 0.035 (residual sugar, density, pH, total sulphur dioxide)

### **4.5.1 Interpretation of Permutation Importance**

Across all permutation runs, a consistent set of features emerged as the strongest contributors to model performance (Table 8). Alcohol exhibited the largest decrease in Cohen’s Kappa upon permutation (0.271), indicating that the model relies heavily on this feature for accurate classification. Sulphates (0.158) and volatile acidity (0.113) followed, reflecting their substantial influence on predicted wine quality. Elevated volatile acidity, in particular, is associated with lower-quality wines, and its importance aligns with established oenological knowledge. Chlorides (0.092) and acidity-related variables showed moderate contributions, while residual sugar, density, pH and total sulphur dioxide had smaller but non-negligible effects. These results suggest that while no single feature fully determines quality, a small subset of chemical attributes consistently drives the model’s predictions.

## **5\. Conclusion** {#5.-conclusion}

This project examined whether physicochemical measurements can be used to predict an ordinal wine-quality tier derived from sensory evaluations, using the red Vinho Verde dataset. To preserve the ordered nature of the target while reducing sparsity, the original quality scores were grouped into four tiers and evaluated through stratified 10-fold cross-validation, with preprocessing performed within each fold to avoid data leakage.

Given the ordinal structure of the problem, model evaluation combined overall accuracy with an asymmetric cost matrix that penalizes severe misclassifications more strongly. This setting better reflects practical scenarios in which confusing distant quality tiers is more harmful than minor errors. Among the evaluated models, tree-based ensembles showed the strongest performance. While Random Forest achieved the lowest total misclassification cost, its tendency to concentrate predictions in majority classes and its failure to detect the lowest-quality tier made it less suitable in operational terms.

Gradient Boosting provided a more balanced solution. After hyperparameter tuning, the selected configuration achieved the highest cross-validated accuracy (68%) while maintaining a competitive misclassification cost and improved sensitivity to minority tiers. Feature relevance analysis based on permutation importance highlighted alcohol as the dominant predictor, followed by sulphates, volatile acidity, and chlorides, confirming that model behavior aligns with known chemical drivers of wine quality.

Overall, the results indicate that laboratory measurements can support early-stage, automated quality assessment, although the observed performance ceiling suggests that chemical attributes alone cannot fully capture perceived wine quality.

### **5.1 Limitations and Improvements** {#5.1-limitations-and-improvements}

***Dataset scope and generalization***  
The analysis is restricted to red Vinho Verde wines, limiting external validity. The learned relationships may not generalize to other wine types or regions without additional validation.

***Label noise and subjectivity***  
Quality tiers are derived from human sensory scores and therefore contain unavoidable subjectivity. The models effectively learn to reproduce this evaluation process rather than an objective notion of quality.

***Class imbalance***  
Despite discretization and stratified validation, extreme quality tiers remain underrepresented. This affects the reliability of predictions for minority classes. Future work could incorporate imbalance-aware training strategies such as class weighting or targeted resampling.

***Interpretability limits***  
Permutation importance captures predictive relevance but does not imply causality and may be affected by correlated features. Complementary interpretability techniques could provide a more robust understanding of model behavior.

***Model Comparison Bias***  
Gradient Boosting benefited from a more focused hyperparameter optimization process compared to Random Forest. Consequently, the comparison may be partially influenced by the depth of tuning rather than model structure alone. It is possible that a heavily tuned Random Forest with class-weighting adjustments could have resolved the sensitivity issues while maintaining its lower cost profile.

## **References** {#references}

\[1\] Basalekou, M., Tataridis, P., Georgakis, K., & Tsintonis, C. (2023). Measuring Wine Quality and Typicity. Beverages, 9(2), 41\. [https\://doi.org/10.3390/beverages9020041](https://doi.org/10.3390/beverages9020041)

\[2\] Wine Quality Dataset (2022). Kaggle link: [https\://www\.kaggle.com/datasets/yasserh/wine-quality-dataset](https://www.kaggle.com/datasets/yasserh/wine-quality-dataset)

\[3\] Thermo Fisher Scientific. (2020). Top six wine   
spoilers \[Brochure\]. [https\://assets.thermofisher.com/TFS-Assets/CMD/brochures/sn-73820-da-top-six-wine-spoilers-sn73820-en.pdf](https://assets.thermofisher.com/TFS-Assets/CMD/brochures/sn-73820-da-top-six-wine-spoilers-sn73820-en.pdf)

[image1]: <data:image/png;base64,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>

[image2]: <data:image/png;base64,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>

[image3]: <data:image/png;base64,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>