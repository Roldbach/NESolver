# NESolver: A Chemistry-Informed Tool for Multivariate Ion Analysis with Ion-Selective Electrodes

> **NESolver: A Chemistry-Informed Tool for Multivariate Ion Analysis with Ion-
> Selective Electrodes**<br>
> Weixun Luo, Martyn G. Boutelle<br>

> **Abstract**<br>
> **Background and Objective:** The vulnerability to cross-ion interference has
> impeded the practical application of ion-selective electrodes. Such cross-
> selectivity is difficult to model and has yet to be fundamentally resolved. We
> aim to develop a computational tool to address the cross-selectivity of ion-
> selective electrodes.<br>
>
> **Methods:** The proposed tool, namely NESolver, is a chemistry-informed tool
> for multivariate ion analysis with ion-selective electrodes. It enables precise
> sensor characteristic quantification and accurate ion concentration inference
> based on the contributions of all present ions in the sample.  The underlying
> computation is enhanced by techniques stemming from theoretical and empirical
> chemical knowledge, ensuring the practical applicability of the tool. A highly
> automated pipeline and an informative text-based user interface have been
> implemented to streamline usage for individuals with various levels of
> technical expertise.<br>
>
> **Results:** We validated the effectiveness of NESolver by benchmarking its
> performance against existing numerical methods on synthetic samples with
> diverse composition, demonstrating its marginal superiority in terms of both
> accuracy and chemical interpretability. By replicating the procedure under
> different experimental conditions, we also showcased the capability of NESolver
> to infer ion concentration beyond the calibration range and robustness to noise.<br>
>
> **Conclusions:** NESolver has shown promising potential as an easy-to-use and
> chemically interpretable tool for multivariate ion analysis, despite its
> current performance limitations on samples with complex composition. Future
> work will focus on enhancing its modelling capability for intricate cross-ion
> interference and validating its performance on real samples.<br>


## Description
- This repository contains the official implementation of NESolver: A Chemistry-
Informed Tool for Multivariate Ion Analysis with Ion-Selective Electrodes. We
have provided our full source code and documentation here.


## Get Started
### Environment Setup
- Please use Anaconda/Miniconda to set up the required virtual environment with
the provided environment configuration file.
    ```
    # 1. Go to the project root directory.
    cd .../NESolver

    # 2. Create the virtual environment with the environment configuration file.
    conda env create -f environment.yml

    # 3. Activate the virtual environment.
    conda activate NESolver

    # 4. Install local packages.
    pip install .
    ```

### Data
- All data used in the experiments is provided in the [data](data) directory.

### Multivariate Ion Analysis
- To run multivariate ion analysis with regression methods, please run the
following command:
    ```
    ./script/multivariate_ion_analysis/run_regression.py  \
        --file_path <file_path>                           \
        --method <method>                                 \
        --training_range <training_range>                 \
        --validation_range <validation_range>             \
        --testing_range <testing_range>                   \
        --seed <seed>                                     \
    ```
    - `--file_path`: A **str** that specifies the file path of data used for the 
    experiment.
    - `--method`: A **str** that specifies the regression method used for the 
    experiment. Please choose one from the following:
        - 'OLS': Ordinary Least Squares Regression
        - 'PLS': Partial Least Squares Regression
        - 'BR': Bayesian Ridge Regression
    - `--training_range`: A **tuple[int, int]** that specifies the range of 
