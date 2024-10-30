This repo contains the submission to the [TrojAI leaderboard](https://pages.nist.gov/trojai/) from IARPA for the round rl-colorful-memory-sep2024. 

Every solution submitted for evaluation is containerized via [Singularity](https://singularity.hpcng.org/) (see this [Singularity tutorial](https://pawseysc.github.io/sc19-containers/)).  

The submitted Singularity container will be run by the TrojAI Evaluation Server using the specified [Container API](https://pages.nist.gov/trojai/docs/submission.html#container-api), inside of a virtual machine which has no network capability.

The container submitted for evaluation must perform trojan detection for a single trained AI model file and output a single probability of the model being poisoned. The test and evaluation infrastructure will iterate over the *N* models for which your container must predict trojan presence. 

The submitted container has access to these [Submission Compute Resources](https://pages.nist.gov/trojai/docs/architecture.html#compute-resources).

--------------
# Table of Contents
1. [Reusing the detector](#reusing-the-detector)
2. [Container Configuration](#container-configuration)
3. [System Requirements](#system-requirements)
4. [Example Data](#example-data)
5. [Submission Instructions](#submission-instructions)
6. [How to Build this Minimal Example](#how-to-build-this-minimal-example)
    1. [Install Anaconda Python](#install-anaconda-python)
    2. [Setup the Conda Environment](#setup-the-conda-environment)
    3. [Test Fake Detector Without Containerization](#test-fake-detector-without-containerization)
    4. [Package Solution into a Singularity Container](#package-solution-into-a-singularity-container)

--------------
# Using the detector

The detector has been mainly coded into the following three files:
* detector.py: File containing the codebase for the detector
* metaparameters.json: The set of tunable parameters used by your container, it should
  validate against metaparameters-schema.json.
* metaparameters-schema.json: JSON schema describing the metaparameters that can be
  changed during inference or training. 
* learned_parameters/: Directory containing data created at training time (that can be 
  changed with re-training the detector)

The variables that change the behavior of the detector are defined in the files below:
* Variables influencing the training of the detector's algorithm: these variables are loaded from the metaparameters.json file and have their name start with "train_". Typically,
these variable are used in the `automatic_configure` and `manual_configure` functions only.
* Training datastructure computed from training variables: these structure are dumped
(in any format) in the learned_parameters folder. During re-training, their content will 
change. These datastructures are created within the `automatic_configure` and 
`manual_configure` functions and should be loaded and used in the `infer` function.
* Inference variables: Similarly to the training variables, variables used only in the
`infer` function are loaded from the metaparameters.json file but start with 
"infer_".

The detector works properly with the provided `entrypoint.py` file and can be packaged in a Singularity container. 
The `entrypoint.py` file should be used as-is and should not be modified.

--------------
# Container Configuration

TrojAI container submissions required that a configuration is included which enables TrojAI T&E to evaluate submitted detectors across various new dimensions. This means that each container needs to: 

- Specify a "metaparameters" file that documents a container's manually tunable parameters and their range of possible values. 
- Generate "learned parameters" via a new reconfiguration API.

Submitted containers will now need to work in two different modes:

- Inference Mode:  Containers will take as input both a "metaparameter" file and a model and output the probability of poisoning. 
- Reconfiguration Mode: Containers will take a new dataset as input and output a file dump of the new learned parameters tuned to that input dataset.

# Container usage: Reconfiguration Mode

Executing the `entrypoint.py` in reconfiguration mode will produce the necessary metadata for your detector and save them into the specified "learned_parameters" directory.

Example usage for one-off reconfiguration:
   ```bash
  python entrypoint.py configure \
  --scratch_dirpath <scratch_dirpath> \
  --metaparameters_filepath <metaparameters_filepath> \
  --schema_filepath <schema_filepath> \
  --learned_parameters_dirpath <learned_params_dirpath> \
  --configure_models_dirpath <configure_models_dirpath>
   ```

Example usage for automatic reconfiguraiton:
   ```bash
   python entrypoint.py configure \
    --automatic_configuration \
    --scratch_dirpath <scratch_dirpath> \
    --metaparameters_filepath <metaparameters_filepath> \
    --schema_filepath <schema_filepath> \
    --learned_parameters_dirpath <learned_params_dirpath> \
    --configure_models_dirpath <configure_models_dirpath>
   ```



# Container usage: Inferencing Mode

Executing the `entrypoint.py` in infernecing mode will output a result file that contains whether the model that is being analyzed is poisoned (1.0) or clean (0.0).

Example usage for inferencing:
   ```bash
   python entrypoint.py infer \
   --model_filepath <model_filepath> \
   --result_filepath <result_filepath> \
   --scratch_dirpath <scratch_dirpath> \
   --round_training_dataset_dirpath <round_training_dirpath> \
   --metaparameters_filepath <metaparameters_filepath> \
   --schema_filepath <schema_filepath> \
   --learned_parameters_dirpath <learned_params_dirpath>
   ```


--------------
# System Requirements

- Linux (tested on Ubuntu 20.04 LTS)
- CUDA capable NVIDIA GPU (tested on A4500)

Note: This example assumes you are running on a version of Linux (like Ubuntu 20.04 LTS) with a CUDA enabled NVIDIA GPU. Singularity only runs natively on Linux, and most Deep Learning libraries are designed for Linux first. While this Conda setup will install the CUDA drivers required to run PyTorch, the CUDA enabled GPU needs to be present on the system. 

--------------
# Example Data

Example data can be downloaded from the NIST [Leader-Board website](https://pages.nist.gov/trojai/). 

A small toy set of clean data is also provided in this repository under the model/example-data/ folder. This toy set of data is only for testing your environment works correctly. 

For some versions of this repository, the example model is too large to check into git. In those cases a model/README.md will point you to where the example model can be downloaded. 

--------------
# Submission Instructions

1. Package your trojan detection solution into a Singularity Container.
    - Name your container file based on which [server](https://pages.nist.gov/trojai/docs/architecture.html) you want to submit to.
2. Request an [Account](https://pages.nist.gov/trojai/docs/accounts.html) on the NIST Test and Evaluation Server.
3. Follow the [Google Drive Submission Instructions](https://pages.nist.gov/trojai/docs/submission.html#container-submission).
4. View job status and results on the [Leader-Board website](https://pages.nist.gov/trojai/).
5. Review your [submission logs](https://pages.nist.gov/trojai/docs/submission.html#output-logs) shared back with your team Google Drive account.


--------------
# How to Build this Minimal Example

## Install Miniforge

[https://conda-forge.org/](https://conda-forge.org/)

## Test Fake Detector Without Containerization

1.  Clone the repository
   
    ```
    git clone https://github.com/usnistgov/trojai-example
    cd trojai-example
    git checkout rl-colorful-memory-sep2024
    ``` 

2. Setup the Conda Environment
   
   - `conda create --name trojai-example python=3.10 -y` ([help](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html))
   - `conda activate trojai-example`
   - `pip install -e .` (Must be run from the trojai-example directory)
   - `pip install torchvision==0.19.1`
   - `pip install tqdm jsonschema jsonpickle` 
   - `pip install scikit-learn==1.5.2 scikit-image==0.24.0`

3. Test the python based `example_trojan_detector` outside of any containerization to confirm pytorch is setup correctly and can utilize the GPU.

    ```bash
    python entrypoint.py infer \
   --model_filepath ./model/rl-colorful-memory-sep2024-example/model.pt \
   --result_filepath ./output.txt \
   --scratch_dirpath ./scratch \
   --round_training_dataset_dirpath /path/to/train-dataset \
   --learned_parameters_dirpath ./learned_parameters \
   --metaparameters_filepath ./metaparameters.json \
   --schema_filepath=./metaparameters_schema.json
    ```

    Example Output:
    
    ```bash
    Trojan Probability: 0.07013004086445151
    ```

4. Test self-configure functionality, note to automatically reconfigure should specify `--automatic_configuration`.

    ```bash
    python entrypoint.py configure \
    --automatic_configuration \
    --scratch_dirpath=./scratch/ \
    --metaparameters_filepath=./metaparameters.json \
    --schema_filepath=./metaparameters_schema.json \
    --learned_parameters_dirpath=./new_learned_parameters/ \
    --configure_models_dirpath=/path/to/new-train-dataset
    ```

    The tuned parameters can then be used in a regular run.

   ```bash
    python entrypoint.py infer \
   --model_filepath ./model/rl-colorful-memory-sep2024-example/model.pt \
   --result_filepath ./output.txt \
   --scratch_dirpath ./scratch \
   --round_training_dataset_dirpath /path/to/train-dataset \
   --learned_parameters_dirpath ./new_learned_parameters \
   --metaparameters_filepath ./metaparameters.json \
   --schema_filepath=./metaparameters_schema.json
    ```

## Package Solution into a Singularity Container

Package `detector.py` into a Singularity container.

1. Install Apptainer
    
    - Follow: [https://apptainer.org/docs/admin/latest/installation.html](https://apptainer.org/docs/admin/latest/installation.html)
        
2. Build singularity container based on `detector.def` file: 

    - delete any old copy of output file if it exists: `rm detector.sif`
    - package container: 
    
      ```bash
      apptainer build detector.sif detector.def
      ```

    which generates a `example_trojan_detector.sif` file.

3. Test run container: 

    ```bash
    apptainer run \
    --bind /full/path/to/trojai-example \
    --nv \
    ./detector.sif \
    infer \
    --model_filepath=./model/rl-colorful-memory-sep2024-example/model.pt \
    --result_filepath=./output.txt \
    --scratch_dirpath=./scratch/ \
    --round_training_dataset_dirpath=/path/to/training/dataset/ \
    --metaparameters_filepath=./metaparameters.json \
    --schema_filepath=./metaparameters_schema.json \
    --learned_parameters_dirpath=./learned_parameters/
    ```

    Example Output:
    ```bash
    Trojan Probability: 0.7091788412534845
    ```
