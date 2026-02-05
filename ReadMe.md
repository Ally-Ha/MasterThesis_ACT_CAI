# Full Run 

## Set Up 
```pip install -r requirements.txt```

## SFT
- Configuration (recipes / SFT / config_pilot.yaml)
- functions Data Processing (scr / data.py)
- Functions Model Processing (scr / model.py)
- Functions SFT (src / sft.py)

Run using: ```script_sft.ipynb```

## Tracking via weights & biases 


# References 

## Sources 
For Reproducibility Reasons, this code is strongly aligned with the proposed SFT, CAI, RFAI from HuggingFaces Alignment Handbook: 

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
