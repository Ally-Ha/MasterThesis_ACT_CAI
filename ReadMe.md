# Therapy-Specific Constitutional AI using ACT Principles
Code supporting the Master Thesis

## Set Up 
```pip install -r requirements.txt```

## Quick Pipeline

### 1. Data generation
- run [script_datageneration.ipynb](script_datageneration.ipynb) which uses [src/data_generation.py](src/data_generation.py) to produce datasets. 
- Main outputs: generic prompts: [recipes/Data_Gen/final/generic_responses_clean.csv](recipes/Data_Gen/final/generic_responses_clean.csv), consitution prompt: [recipes/Data_Gen/final/constitution_responses_clean.csv](recipes/Data_Gen/final/constitution_responses_clean.csv), and [recipes/Data_Gen/final/ds_constitution](recipes/Data_Gen/final/ds_constitution).

### 2. Critique & Revision (Constitutional AI)
- run [script_critique_revise.ipynb](script_critique_revise.ipynb) which calls [src/critique_revision.py](src/critique_revision.py) to critique and revise the constitution responses. 
- Main outputs: [recipes/Data_Gen/final/constitution_revised.csv](recipes/Data_Gen/final/constitution_revised.csv) and [recipes/Data_Gen/final/ds_constitution_revised](recipes/Data_Gen/final/ds_constitution_revised).

### 3. Supervised Fine-Tuning (SFT) 
- run [script_sft.ipynb](script_sft.ipynb) which uses [src/sft.py](src/sft.py) and the model configs in [recipes/SFT/](recipes/SFT/) to train. 
- Main outputs: model checkpoints and training logs in the configured `output_dir` (see [recipes/SFT/config_model0.yaml](recipes/SFT/config_model0.yaml)).

### 4. Evaluation & Analysis
- run [script_eval.ipynb](script_eval.ipynb) which uses [src/eval.py](src/eval.py) and [src/questionnaires.py](src/questionnaires.py). 
- Main outputs: evaluation logs and reports in `logs/eval`.


## References 
For Reproducibility Reasons, this code is strongly aligned with the proposed SFT, CAI, RFAI from the following ressources:  

```bibtex
@software{Tunstall_The_Alignment_Handbook,
  author = {Tunstall, Lewis and Beeching, Edward and Lambert, Nathan and Rajani, Nazneen and Huang, Shengyi and Rasul, Kashif and Bartolome, Alvaro and M. Patiño, Carlos and M. Rush, Alexander and Wolf, Thomas},
  license = {Apache-2.0},
  title = {{The Alignment Handbook}},
  url = {https://github.com/huggingface/alignment-handbook},
  version = {0.4.0.dev0}
}

@online{Labonne2024SFTLlama,
  author       = {Labonne, Maxime},
  title        = {Fine-tune Llama 3.1 Ultra-Efficiently with Unsloth},
  year         = {2024},
  month        = jul,
  day          = {29},
  url          = {https://huggingface.co/blog/mlabonne/sft-llama3},
  note         = {Hugging Face Blog}
}

@article{Huang2024cai,
  author = {Huang, Shengyi and Tunstall, Lewis and Beeching, Edward and von Werra, Leandro and Sanseviero, Omar and Rasul, Kashif and Wolf, Thomas},
  title = {Constitutional AI Recipe},
  journal = {Hugging Face Blog},
  year = {2024},
  note = {https://huggingface.co/blog/constitutional_ai},
  GitHub = {https://github.com/huggingface/llm-swarm},
}

@online{UbiAI2024PsychLlama,
  author       = {{UbiAI}},
  title        = {Fine-Tuning LLaMA-3 for Psychology Question Answering Using LoRA and Unsloth},
  year         = {2024},
  month        = dec,
  day          = {5},
  url          = {https://ubiai.tools/fine-tune-llama-3-psychology-question-unsloth/},
  note         = {UbiAI Blog}
}
```
