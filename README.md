<div align="center">
  <h1 align="center">Calibration of Continual Learning Models</h1>
</div>

This repository provides code for the experiments done in the paper [Calibration of Continual Learning Models](https://en.wikipedia.org/wiki/Placeholder). 

## Citation
```
@misc{wiki:Placeholder,
   author = "Wikipedia",
   title = "{Placeholder} --- {W}ikipedia{,} The Free Encyclopedia",
   year = "2024",
   howpublished = {\url{http://en.wikipedia.org/w/index.php?title=Placeholder&oldid=1085025565}},
   note = "[Online; accessed 09-April-2024]"
 }
```

## Repository structure
```
📦Continual-Calibration
 ┣ 📂example 
 ┃ ┣ 📜SplitMNIST_metrics_plots.ipynb   # Sample notebook for plotting metrics
 ┃ ┗ 📜SplitMNIST_script.sh   # Example script
 ┣ 📜.gitignore
 ┣ 📜Continual_Calibration.py 
 ┣ 📜DQN_model.py
 ┣ 📜ECE_metrics.py
 ┣ 📜Ent_Loss.py
 ┣ 📜ModelDecorator.py   # Model decorators for post-calibration(TS/VS/MS)
 ┣ 📜README.md
 ┣ 📜ResNet18.py
 ┣ 📜atari_dataset.py
 ┗ 📜main.py
```

## Getting Started

### 1. Clone the repository
```shell
git clone https://github.com/lilanpei/Continual-Calibration.git
cd Continual-Calibration
```
### 2. Creat an environment from the environment.yml file
```shell
conda env create -f Continual-Calibration/environment.yml
```
### 3. Activate the environment
```shell
conda activate continual-calibration
```
### 4. Run the example script
```shell
./example/SplitMNIST_script.sh
```   

### 5. Plot the results from 
```
SplitMNIST_metrics_plots.ipynb
```
