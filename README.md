<h1>Wine Quality Prediction</h1>

<h3>Project Background</h3>

<p>This analysis focuses on a Portuguese wine producer producing red Vinho Verde wines. Assessing wine quality traditionally relies heavily on manual sensory evaluations by expert tasters. This process is both time-consuming and expensive. The primary business objective is to evaluate whether routine, objective physicochemical laboratory tests can be leveraged to predict sensory quality tiers, thereby enabling cost-effective, early-stage quality estimation and supporting inventory/pricing decisions.</p>

<p>Insights and recommendations are provided on the following key areas:</p>
<ul>
  <li>
    Key Chemical Quality Drivers: Identification of the physicochemical attributes (e.g., alcohol content, volatile acidity, sulphates) that heavily influence         perceived quality.</li>
  <li>
    Cost-Sensitive Model Performance: Evaluating machine learning classifiers under an asymmetric cost matrix that heavily penalizes critical business                 misclassifications (e.g., mislabeling an Excellent wine as Poor).
  </li><li>
    Class Imbalance & Minority Detection: Analyzing performance trade-offs between standard accuracy and the model's sensitivity in detecting rare, high-risk, or      premium quality tiers.
  </li><li>
    Operational Strategy & Implementation: Actionable recommendations for quality control teams, early-stage sorting, and cellar operations.
  </li>   
</ul>

<h3>Data Structure & Initial Checks</h3>

<p>The company's primary dataset consists of a single relational dataset containing a total row count of 1,143 records of red Vinho Verde wines. The primary target variable is quality (an ordinal sensory score ranging from 0 to 10), accompanied by 11 numeric physicochemical feature columns.</p>

<p>A description of the core feature categories and tables is as follows:</p>
<ul>
  <li>Acidity & pH Variables (fixed acidity, volatile acidity, citric acid, pH): Defines the chemical acidity profile impacting taste balance, freshness, and        microbial stability. </li><li>  
  Stability & Preservation Additives (free sulfur dioxide, total sulfur dioxide, sulphates): Compounds used to prevent oxidation and microbial growth, directly      influencing shelf life and aroma retention.</li><li>   
  Body & Composition Features (residual sugar, chlorides, density, alcohol): Dictates mouthfeel, structural balance, saltiness, and overall body perception.         </li> <li>Target Variable & Metadata (id, quality): id serves as a unique identifier (removed during feature selection); quality is the target sensory rating      discretized into 4 business tiers (Poor, Below Average, Above Average, Excellent).</li>   
</ul>

| Attribute Group | Attributes | Description / Role |
| :--- | :--- | :--- |
| **Identifier** | `id` | Primary Key (PK, removed from modeling) |
| **Acidity & pH** | `fixed_acidity`, `volatile_acidity`, `citric_acid`, `pH` | Chemical measurements for acidity profile |
| **Sulfur & Sulfites** | `free_sulfur_dioxide`, `total_sulfur_dioxide`, `sulphates` | Preservative and chemical compound indicators |
| **Composition & Density** | `residual_sugar`, `chlorides`, `density`, `alcohol` | Physical and chemical concentration features |
| **Target Variable** | `quality` | Target feature (Discretized Tiers) |

<h3>Executive Summary</h3>

<h4>Overview of Findings</h4>
<p>
Laboratory physicochemical measurements can successfully guide early-stage quality tiering, with Alcohol content ($\Delta\text{Kappa} = 0.271$), Sulphates ($0.158$), and Volatile Acidity ($0.113$) serving as the primary drivers of wine quality. While standard Random Forest models achieved slightly higher raw accuracy (67.6%) and lower baseline misclassification cost (2,932), they suffered from a critical operational flaw by failing entirely to identify Poor quality wines (0% recall). Hyperparameter-tuned Gradient Boosting (4-depth, 200 trees, 0.05 learning rate) was selected as the optimal model, achieving 68% accuracy while maintaining essential sensitivity to minority quality tiers and minimizing severe misclassification risk.   
</p>

<h4>Summary table:</h4>

| Model | Accuracy | Cost |
| :--- | :---: | :---: |
| **Gradient Boosting (Fine Tuned)** | 68.0% | 3113 |
| **Random Forest** | 67.6% | 2932 |
| **MLP** | 61.2% | 4091 |
| **SVM** | 56.7% | 4453 |
| **KNN** | 51.3% | 5964 |

<h3>Insights Deep Dive</h3>
<h4>Key Chemical Quality Drivers:</h4>
<ul>
  <li>Alcohol Content is the strongest positive quality indicator. Statistical correlation ($r = +0.48$) and permutation testing ($\Delta\text{Kappa} = 0.271$)      demonstrate that higher alcohol levels consistently correlate with superior quality ratings.</li><li>Volatile Acidity acts as a major negative quality             indicator. Elevated acetic acid ($r = -0.41$, $\Delta\text{Kappa} = 0.113$) imparts undesirable vinegar-like off-notes, heavily driving down sensory scores.       </li><li>Sulphates play a key secondary stabilization role. Sulphates ($\Delta\text{Kappa} = 0.158$) serve as an essential antioxidant and antimicrobial agent,    directly protecting taste profile integrity.</li><li>Minor impact features. Residual sugar, density, pH, and total sulfur dioxide demonstrated minimal direct      predictive power ($\Delta\text{Kappa} < 0.035$) in tier differentiation.</li>   
</ul>

| Attribute | $\Delta\text{Kappa}$ |
| :--- | :---: |
| **Alcohol** | 0.271 |
| **Sulphates** | 0.158 |
| **Volatile Acidity** | 0.113 |
| **Chlorides** | 0.092 |
| **Fixed Acidity** | 0.047 |
| **Citric Acid** | 0.047 |
| **Free Sulphur Dioxide** | 0.038 |

*Note: Permutation Importance excluding attributes with ΔKappa < 0.035 (residual sugar,density, pH, total sulphur dioxide)
<p></p>
<h4>Cost-Sensitive Model Performance:</h4>
<ul>
  <li>Asymmetric risk profile defined for business impact. Failing to identify a premium product (Excellent misclassified as Poor) was penalized heavily (cost =     150), whereas overestimating low quality carries lower penalties (cost = 100). Adjacent tier errors carried low penalties (4 to 10).</li><li>Baseline algorithms   proved insufficient. Simple classifiers like KNN (51.3% accuracy, 5,964 cost) and SVM (56.7% accuracy, 4,453 cost) produced severe errors due to overlapping       feature distributions.</li><li>Neural Networks underperformed tree ensembles. Multi-Layer Perceptron (MLP) achieved 61.2% accuracy and 4,091 cost, failing to      justify its complexity compared to tree ensembles.</li><li>Gradient Boosting achieved superior operational safety. While Random Forest yielded a lower cost        metric (2,932), tuned Gradient Boosting provided the best balance of total accuracy (67.5%–68.0%) and risk-managed misclassifications.</li>   
</ul>

<h4>Class Imbalance & Minority Detection:</h4>
<ul>
  <li>Severe target skewness in raw data. Extreme quality scores (3–4 and 7–8) were heavily underrepresented, necessitating 4-tier discretization (Poor, Below       Average, Above Average, Excellent).</li><li>Random Forest failed minority detection. Despite strong top-line metrics, Random Forest recorded a 0% recall on the    Poor quality tier (0 out of 39 identified), rendering it unacceptable for business deployment.</li><li>Gradient Boosting captured extreme tiers. Gradient          Boosting successfully identified minority instances across both Poor and Excellent tiers, supporting comprehensive quality monitoring.</li><li>Stratified cross-   validation validated robustness. Stratified 10-fold cross-validation ensured fold-level class proportions were preserved and scaling parameters were isolated to   prevent data leakage.</li>   
</ul>

<h4>Hyperparameter Tuning & Model Optimization:</h4>
<ul>
  <li>Tree Depth of 4 captured complex non-linear boundaries. Shallow depth models (depth 2 and 3) exhibited underfitting (max accuracy ~62.9–65.5%).</li><li>   
  Optimal learning rate set at 0.05. Lower learning rates (0.05) provided stable convergence and better generalization compared to aggressive settings (0.2).</li>   <li>Ensemble size optimized at 200 trees. Expanding the model to 200 trees at depth 4 optimized performance (68.0% accuracy, 3,133 cost) with diminishing          returns beyond 100 trees.</li><li>Tuning overcame initial baseline limitations. Hyperparameter tuning allowed Gradient Boosting to surpass Random Forest in        accuracy while maintaining non-zero recall on minority classes.</li>   
</ul>

<h4>Gradient Boosting Hyperparameters:</h4>

<table>
  <tr>
    <td align="center"><b>Comparison of Total Cost to Depth versus Number of Trees for Learning Rate of 0.1</b></td>
    <td align="center"><b>Comparison of Total Cost to Depth versus Number of Trees for Learning Rate of 0.2</b></td>
    <td align="center"><b>Comparison of Total Cost to Depth versus Number of Trees for Learning Rate of 0.05</b></td>
  </tr>
  <tr>
    <td><img width="400" height="380" alt="table2" src="https://github.com/user-attachments/assets/7cb58155-3762-45ea-b520-3c80b888e0ee" /></td>
    <td><img width="400" height="380" alt="table3" src="https://github.com/user-attachments/assets/b67992da-5968-4eac-91ec-78a34f4c3110" /></td>
    <td><img width="400" height="380" alt="table1" src="https://github.com/user-attachments/assets/8409c1c4-d748-4b6d-86d2-36a866a00ac7" /></td>
  </tr>
</table>

<h3>Recommendations:</h3>

<p>Based on the insights and findings above, we recommend the Quality Assurance, Production, and Operations Teams consider the following:</p>
<ul>
  <li>Volatile acidity strongly correlates with low quality ratings due to off-flavors ($r = -0.41$). Implement automated batch alerts during fermentation when      volatile acidity metrics breach pre-set thresholds to allow early corrective intervention.</li><li>   Alcohol content and sulphate concentrations are the          dominant positive drivers of quality ($\Delta\text{Kappa}$ of 0.271 and 0.158). Optimize pre-fermentation sugar/grape selection and monitor sulphate additions     to preserve sensory structure for premium product lines.</li><li>   Random Forest models mask operational risk by achieving high accuracy through total            rejection of minority classes (Poor tier recall = 0). Reject pure accuracy metrics for model deployment and enforce cost-sensitive Gradient Boosting models that   maintain sensitivity across all quality tiers.</li><li>   Chemical analysis alone reaches a performance ceiling around 68% accuracy due to subjective human        taster evaluation. Position the ML model as an automated early-stage screening tool to pre-sort batches rather than replacing expert sensory panels entirely.      </li><li>Adjacent tier misclassifications carry low operational impact, whereas extreme tier misclassifications create major financial losses. Incorporate the     custom asymmetric cost matrix directly into future model retrain loops to continuously minimize high-penalty errors.</li>   
</ul>

<h3>Assumptions and Caveats:</h3>

<p>Throughout the analysis, multiple assumptions were made to manage challenges with the data:</p>
<ul>
  <li>Regional Specificity: The dataset is restricted exclusively to red Vinho Verde wines; insights, correlations, and model weights may not directly generalize     to white wines or other geographical regions.</li><li>Subjective Target Noise: Quality tiers are derived from human sensory panels, which inherently possess       subjective bias and label noise that purely chemical metrics cannot fully reconcile.</li><li>Quality Discretization Mapping: Quality ratings were discretized      into four categories (Poor: 3–4, Below Average: 5, Above Average: 6, Excellent: 7–8) using rule engines to handle extreme class sparsity.</li><li> 
  Identifier Feature Exclusion: The unique identifier column (id) was dropped under the assumption that it contained zero predictive power and would introduce        spurious correlation.</li><li>Outlier Retention & Normalization: Highly skewed features like chlorides (skew = 6.03) and residual sugar (skew = 4.4) contained     extreme values that were retained to reflect real batch variations and normalized via Z-score scaling within CV loops.</li>  
</ul>
