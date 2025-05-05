Setting up environments:

    - Install the suitable miniconda environment here using the following commands:
        - Download [miniconda](https://repo.anaconda.com/miniconda/Miniconda3-py311_24.5.0-0-Linux-x86_64.sh)
        - Install the miniconda version using this command ``bash Miniconda3-py311_24.5.0-0-Linux-x86_64.sh``.
        - Under miniconda and install conda environment:
        - ```${conda_env_path}``` is the directory to install conda environment
        ```bash
            source ${conda_env_path}/etc/profile.d/conda.sh
        ```
    - Create environments for different tasks:
        There are two options to set up the required environment:
        - Option one: Using yaml files
        For SLA, ASR and GEC and all scoring-tools:
        ```bash
            conda env create -f envs/sandi-all.yaml --prefix ${conda_env_path}/envs/${environment_name_1}
            conda activate ${environment_name_1}
            python -m spacy download en_core_web_sm
            pip install numpy==1.24.3    
        ```
        * ignore error messages for dependency conflicts on blis 1.0.1 and thinc 8.3.2

        For DD:
        ```bash
            conda env create -f envs/sandi-dd.yaml --prefix ${conda_env_path}/envs/${environment_name_2}
        ```

        - Option two: Using required package lists
        For SLA, ASR and GEC and all scoring tools:
        ```bash
            conda create --name ${environment_name_1} python=3.10
            conda activate ${environment_name_1}
            pip install -r envs/sandi-all.txt
            python -m spacy download en_core_web_sm
            pip install numpy==1.24.3    
        ```
        * ignore error messages for dependency conflicts on blis 1.0.1 and thinc 8.3.2

        For DD:
        ```bash
            conda create --name ${environment_name_2} python=3.8
            conda activate ${environment_name_2}
            pip install -r envs/sandi-dd.txt
        ```

    - Update the paths in ```envs/sandi-all-env.sh``` and ```envs/sandi-dd-env.sh``` to your locations
