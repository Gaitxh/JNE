# Jacobian-Based Interpretation of Nonlinear Neural Encoding Model

Our paper has been accepted by NeurIPS2025 as **Spotlight** [https://arxiv.org/abs/2510.13688].

### Background and Motivation
![Figure1](https://github.com/user-attachments/assets/041b88f9-0e46-4733-9c8d-269679f1ce70)

Recent advancements in neural encoding models have aligned artificial neural networks (ANNs) with functional magnetic resonance imaging (fMRI), enhancing research on neural representation and interpretability. While many studies show that neural responses are nonlinear, the quantitative characterization of this nonlinearity and its cortical distribution remains largely unknown. Simply comparing the performance of linear and nonlinear encoding models does not effectively capture the nonlinear characteristics of BOLD responses (Fig. b). To address this issue, we propose a new nonlinear assessment metric based on the Jacobian matrix (Fig. c).
![Figure2_2](https://github.com/user-attachments/assets/af1f024a-6552-4692-9400-5965662ef05c)


### Framework and Kernel Concept

We introduce a new tool grounded in the concept of the Jacobian, which reflects local perturbations within the system. Specifically, the Jacobian Nonlinearity Estimator (JNE) serves as an interpretability metric for nonlinear neural encoding models. It quantifies nonlinearity by assessing the statistical dispersion of the input-output Jacobian matrix, thereby approximating the degree of nonlinearity in voxel-level BOLD responses. By leveraging this framework, we aim to enhance the understanding of neural encoding dynamics and provide a robust mechanism for interpreting the complex interactions inherent in neural representations. This approach not only facilitates the analysis of nonlinear responses but also contributes to a deeper comprehension of the underlying neural processes.

### Highlights

• Validated JNE via simulations (Result Ⅰ), showing its ability to quantify nonlinearity across single activation functions (Result Ⅰ-1) and architectures (Result Ⅰ-2). Deeper nonlinear configurations yielded higher JNE values, with activation invariant hierarchical trends.

• Primary visual cortices(e.g.,V1) exhibit lower BOLD response nonlinearity(smaller JNE values), while higherorder visual cortices(e.g.,FFA,PPA) show stronger nonlinear characteristics(larger JNE values), indicating functional specificity in the spatial distribution of nonlinear responses(Result Ⅱ).

• Nonlinear responses in the visual cortex show a hierarchical progression(Result Ⅲ):JNE values increase sequentially from primary visual cortex(V1) to intermediate visual cortex(V2-V4) and further to higher-order visual cortex(FFA,EBA,etc.), consistent with established hierarchical cortical organization.

• The extended Sample-Specific JNE(JNE-SS) reveals stimulus selective nonlinear response patterns in functionally specialized brain regions(Result Ⅳ), e.g., PPA and RSC show stronger nonlinear responses to outdoor scenes, while FFA exhibits heightened nonlinearity for face stimuli.

### Results
#### I-Simulation-Based Validation of JNE
![Figure4_prior](https://github.com/user-attachments/assets/b0b148e7-adac-49fb-b551-1aa884a5b86e)

#### II-Spatial Distribution of BOLD Response Nonlinearity in Visual Cortex
![Figure4](https://github.com/user-attachments/assets/48cd457b-fcb5-450e-ac3b-0aca444e92e4)

#### III-Hierarchical Progression of Nonlinearity Across the Visual Cortex
![Figure5](https://github.com/user-attachments/assets/6f343da8-13a4-45ab-bfae-2d9049023dca)

#### IV-Sample Selectivity of BOLD Response Nonlinearity
![Figure6](https://github.com/user-attachments/assets/7465b950-b75b-4263-b716-cd0ebb80bc27)

#### If you have any other requirements, please contact gaitxh@foxmail.com and we will do our best to meet them.
