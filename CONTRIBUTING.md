# Set up instructions for contributors

### 1. Clone this repo with submodules

```
git clone --recursive https://github.com/wenxindongwork/keras-tuner-alpha.git
```

If already cloned, add the submodules

```
git submodule update --init --recursive
```

**Troubleshooting**:

If you don't see the MaxText repository after cloning or updating, try

```
git submodule add --force https://github.com/google/maxtext
```

### 2. Install dependencies on a TPU or GPU VM

Kithara requires `Python>=3.11`.

1. With `conda`: 
    ```
    conda create -n kithara_env python=3.11
    conda activate kithara_env

    # On TPU
    pip install -e .[tpu,dev] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html -f https://download.pytorch.org/whl/cpu 

    # On GPU
    pip install -e .[gpu,dev] -f https://download.pytorch.org/whl/cpu 
    ```
2. With `venv`:

    ```
    python3.11 -m venv kithara_env
    source kithara_env/bin/activate 
    
    # On TPU
    pip install -e .[tpu,dev] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html -f https://download.pytorch.org/whl/cpu 

    # On GPU
    pip install -e .[gpu,dev] -f https://download.pytorch.org/whl/cpu 
    ```

# Releasing a New Version of Kithara (Googlers Only)
These instructions are for Googlers with access to Kithara's credentials via go/valentine.  Search for "kithara" on go/valentine to retrieve the necessary credentials for TestPyPI and PyPI.

**Before you begin**: Ensure you have the latest code and have thoroughly tested your changes.

### Release Steps

1. **Run Tests**:  Execute all tests and confirm they pass successfully. This will take a while. :

    ```
    python -m unittest
    ``` 

2. **Update Version Number**: Modify the `pyproject.toml` file, incrementing the version number according to the MAJOR.MINOR.PATCH convention.  For example, if releasing version 1.2.3, and the next release is a patch, change it to 1.2.4.


3. **Build the Wheel File**: Create the distribution wheel file. After running the following command, you should see a `.whl` file and a `.tar.gz` file created under `/dist`. 

    ```
    flit build --no-use-vcs
    ```

4. **Upload to TestPyPI**. 

    - Configure .pypirc: Create (or update) the `$HOME/.pypirc` file with the TestPyPI credentials found in go/valentine. This file tells twine where and how to upload.


    - Upload the newly built wheel file to TestPyPI:

        ```
        twine upload --repository testpypi dist/*
        ```

5. **Test on a Fresh TPU VM**:  

    - **Set up Conda**: If Conda isn't already installed, follow these steps:

        ```
        mkdir -p ~/miniconda3
        wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
        bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
        rm ~/miniconda3/miniconda.sh
        ```
        After installing Conda, close and reopen your terminal application or refresh it by running the following command:
        ```
        source ~/miniconda3/bin/activate
        ```

        To initialize conda on all available shells, run the following command:
        ```
        conda init --all
        ```

    - **Create and Activate Conda Environment**:

        ```
        conda create -n test python==3.11
        conda activate test
        ```
    - **Install Kithara from TestPyPI**: This command installs Kithara and its dependencies. It prioritizes TestPyPI for finding packages but will fall back to PyPI. Please replace VERSION with the release version.
        ```
        pip install kithara[tpu]==VERSION --index-url https://pypi.org/simple --extra-index-url https://test.pypi.org/simple/ -f https://storage.googleapis.com/jax-releases/libtpu_releases.html --extra-index-url  https://download.pytorch.org/whl/cpu --force-reinstall
        ```
    - **Resolve Dependency Conflicts (Important)**: Due to the mixed PyPI/TestPyPI installation, you might encounter dependency conflicts.  Currently, the following packages need to be specifically reinstalled from PyPI:

        ```
        pip uninstall ml-goodput-measurement -y && pip install ml-goodput-measurement --index-url https://pypi.org/simple 

        pip uninstall chex -y && pip install chex>=0.1.88 --index-url https://pypi.org/simple 
        ```
        _Note: This list may change.  If you encounter other issues, compare package versions with `pip show pacakge_name` between this new environment and your own development environment and reinstall any conflicting packages from PyPI._

6. **Run Single-Host End-to-End Examples**: Clone the Kithara repository (using the correct branch) and execute the tests to make sure everything works as expected.

    ```
    git clone -b BRANCH_NAME https://github.com/wenxindongwork/keras-tuner-alpha.git
    ```
    Remove the 
    ```kithara``` source code folder just in case. 
    ```
    cd keras-tuner-alpha
    rm -rf kithara
    ```
    Log into HuggingFace with your own API token 
    ```
    huggingface-cli login
    ```
    Run the set of light tests. 
    ```
    RUN_LIGHT_TESTS_ONLY=1 python -m unittest
    ```
    Run the e2e tests. 
    ```
    python -m unittest tests/trainer/test_sft_e2e.py
    ```
    
7. **Run Multi-Host End-to-End Examples**: 


8. **Upload to PyPI**: Once testing on TestPyPI is successful, you'll need to upload the wheel to the official PyPI repository. 

    ```
    twine upload --repository pypi dist/* 
    ```
    

