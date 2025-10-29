# Setting up a Conda Environment for Transformers

Here's the extracted and organized information on how to set up a Conda environment for a transformers environment.

## Step 1: Create a new conda environment

*   Create a new conda environment with Python 3.10:
    ```bash
    conda create -n myenv python=3.10
    ```
*   Activate the newly created environment:
    ```bash
    conda activate myenv
    ```

## Step 2: Install packages

*   Install PyTorch, TorchVision, and TorchAudio from the pytorch channel:
    ```bash
    conda install pytorch torchvision torchaudio -c pytorch
    ```
*   Install Hugging Face Transformers from the huggingface channel:
    ```bash
    conda install -c huggingface transformers
    ```
*   Install scikit-learn and pandas from the conda-forge channel:
    ```bash
    conda install -c conda-forge scikit-learn pandas
    ```

## Step 3: Install and activate ipykernel

*   Install ipykernel from the conda-forge channel:
    ```bash
    conda install -c conda-forge ipykernel
    ```
*   Activate the new kernel for your notebook:
    ```bash
    python -m ipykernel install --user --name=myenv
    ```

## After Setup:

*   You can now select your new kernel when running your notebook.
*   To deactivate your environment:
    ```bash
    conda deactivate
    ```
*   To reactivate your environment:
    ```bash
    conda activate myenv
    ```
*   To remove an environment:
    ```bash
    conda env remove --name myenv
    ```
