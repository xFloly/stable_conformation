# Stable Conformation Probability Analysis (Course Project)

This repository contains code and experiments for **Stable Conformation Probability Analysis**, a course project in machine learning for drug design. The project investigates how effectively a flow-based generative model can evaluate and predict the stability of molecular conformations.

We build on the [Continuous Graph Conformation Flow (CGCF)](https://github.com/DeepGraphLearning/CGCF) model by Xu *et al.* (ICLR 2021), using a pre-trained CGCF (trained on the [GEOM](https://github.com/learningmatter-mit/geom) dataset of molecular conformations) as a normalizing flow to assign probabilities (log-likelihoods) to different 3D conformations of molecules.

To evaluate the model's ability to recognize stable structures, we simulate suboptimal conformers through controlled perturbations of atom positions and molecular orientations. We then compare the CGCF model’s probability estimates against classical methods such as MMFF (Merck Molecular Force Field) optimizations. As a primary metric of deviation, we use **Root-Mean-Square Deviation (RMSD)** to quantify how far a perturbed conformation strays from a known stable reference structure.

## Repository Structure

The repository is organized into the following main components:

- **`model/`** – Contains the code for the CGCF normalizing flow model, adapted for this project. This includes utilities needed to load the pretrained model and evaluate log-probabilities of conformations. IGNACY

- **`datasets/`** - Code and data utilities for handling molecular conformations used in our
experiments. This may include scripts to generate or load molecular structures and their
conformers.


- **`analyse_perturb.py`** Tests the EdgeCNF model’s sensitivity by perturbing atomic coordinates with different noise levels. Scores the original and progressively noisier conformations, demonstrating how log-likelihoods change with increasing structural distortion.

- **`analyse_from_sdf.py`** Evaluates log-likelihoods for original and generated conformations from SDF files using the pretrained EdgeCNF model. Ranks the original structure, computes RMSD and correlation metrics, and summarizes model performance across molecules.

- **`correlation_analysis.ipynb`** – Jupyter Notebook containing the statistical analysis and
visualizations for our experiments. This notebook aggregates results (e.g., from the above scripts)
and produces 

## Methodology and Hypotheses

### Data Perturbation Methodology

In the absence of "bad" conformations in the provided data (the CIF crystal structures), we generate **synthetic suboptimal conformations** to challenge the model. Starting from a known correct (low-energy) conformation of each molecule, we introduce controlled random perturbations:

- **Atomic positional noise**: Small random displacements are added to the coordinates of each atom. This simulates slight deviations in bond lengths and angles.
- **Orientation perturbation**: The entire molecule is randomly rotated and translated before evaluation. *(While RMSD calculations typically align structures to cancel out overall rotations/translations, we still randomize orientation to ensure the model’s evaluation is orientation-invariant and to diversify initial states for force-field optimization.)*

![RMSD-vs-logP-perturb](images/rmsd_logp_random_noise.png)
- **Generating more realistic conformations**:  
  For generating more realistic conformations, we use:
  `rdkit.Chem.rdDistGeom.EmbedMultipleConfs`,  
  which employs Experimental-Torsion Knowledge Distance Geometry (ETKDG). This method incorporates torsion angle preferences derived from the Cambridge Structural Database (CSD), resulting in physically plausible 3D geometries.
![RMSD-vs-logP-etkdg](images/rmsd_logp_rdkit_etkdg.png)
We generate a range of perturbed conformers per molecule, from near-native (very small RMSD) to heavily distorted (large RMSD). **RMSD (Root Mean Square Deviation)** is computed after aligning each perturbed conformation to the reference structure and serves as a quantitative measure of distortion — higher RMSD indicates a more significant departure from the stable conformation.




Using these perturbed conformers, their CGCF log-probabilities, and classical optimization results, we explore the following research hypotheses:

---

### Hypothesis 1: Local Minima Detection for OOD Conformations

A flow-based model can recognize and effectively locate local energy minima even for **out-of-distribution (OOD)** conformations.

Even if a conformation is far from anything seen in training (e.g., a highly distorted structure), the model’s probability evaluation should guide us toward a nearby stable state. We expect that conformers closer to true low-energy structures (smaller RMSD) will receive higher log-probability scores, indicating the model’s ability to detect when a molecule is in a basin of attraction around a local minimum.

We evaluate this both on randomly noised conformations and on realistic conformations generated using RDKit’s ETKDG method.

<img src="images/hypothesis1.png" alt="LogP-values-vs-RMSD" width="300"/>

---

### Hypothesis 2: Energy–Log Probability Correlation

There is a meaningful correlation between a molecule’s physically optimized energy and the CGCF model’s **log-probability** for that conformation.

Since CGCF was trained on ensembles of low-energy conformers, we hypothesize that its learned probability distribution reflects true molecular stability. We test this by plotting conformational energies (e.g., from MMFF or DFT) against the model-computed log-probabilities.

A Spearman inverse correlation would support the idea that the flow’s probability landscape approximates the true energy surface — i.e., lower-energy conformers should have higher probabilities.

Our analysis shows trends indicating this relationship, although not perfectly — likely due to the model being trained on likelihood, not energy, directly.
Few examples:

| SMILES | Spearman | Pearson |
|--------|----------|---------|
| `CO[C@H]1CC[C@H]1O[CH][NH]` | -0.6321 (p = 0.0115) | -0.7790 (p = 0.0006) |
| `O=C1CC(=O)[C@@H](O)CO1` | -0.1905 (p = 0.6514) | -0.2597 (p = 0.5346) |
| `C[C@@H](O)C#C[C@H]1C[C@H]1C` | -0.4939 (p = 0.0014) | -0.3203 (p = 0.0468) |


## Setup and Usage

To reproduce the experiments and use the code in this repository, follow these steps:

### 1. Clone the Repository

```bash
git clone https://github.com/xFloly/stable_conformation.git
cd stable_conformation
```

### 2. Create the Conda Environment
Ensure you have Anaconda or Miniconda installed. Use the provided `environment.yaml` file to set up the environment with all required dependencies:
```bash
conda env create -f environment.yaml
conda activate stable_conf
```

### 3. Obtain Pre-trained CGCF Model Weights
The CGCF model used here should be pre-trained on the GEOM dataset for optimal results. If the pretrained checkpoint is not included in this repository (due to file size), you will need to obtain it yourself:

Download a suitable checkpoint from the original [CGCF](https://github.com/MinkaiXu/CGCF-ConfGen) repository or project page. 

Alternatively, you may train a CGCF model on the GEOM dataset yourself (note that this is time-consuming and not the focus of this project). 

Once you have the model file (e.g., `ckpt_drugs.pt`), place it in an appropriate location — for example:  IGNACY - zweryfikuj

```bash
model/CGCF/ckpt_geom.pt
```
Or specify the path in the configuration when running scripts. The code will attempt to load this checkpoint automatically.

### 4. Calcualte log-probability 
IGNACY - dopisz jak to odpalałeś

For our course experiments and data analysis, we used the dataset located [here](somelink).
