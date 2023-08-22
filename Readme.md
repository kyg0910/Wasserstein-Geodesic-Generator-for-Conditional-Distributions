# Introduction
This supplementary file provides the implementation used for the results in Wasserstein Geodesic Generator for Conditional Distributions" [[paper]](https://arxiv.org/abs/2308.10145)

<table>
<tr>
<td>
<img src="fig/fig_generation.png" width="100%" alt="Results by the proposed method">
</td>
<td>
<img src="fig/generation_cAAE.png" width="100%" alt="Results by the baseline method">
</td>
</tr>
<tr>
<td colspan="2">

**Conditional generation results by the proposed method (left) and cAAE (right).** The proposed method produces face images with clearer eyes, noses, and mouths than baselines. For each method, the leftmost and rightmost columns show generation results for observed domains and intermediate columns show results for unobserved intermediate domains.


</td>
</tr>
</table>

<table>
<tr>
<td>
<img src="fig/fig_translation.jpg" width="70%" alt="A visualization of latent interpolation with the real data">
</td>
</tr>
<tr>
<td>

**A visualization of latent interpolation with the real data and their translation results by various methods.** The bottom row shows real images from two observed domains. From the first to third rows, the leftmost column shows the ground-truth, the rightmost column shows transportation results of the ground-truth, and intermediate columns show latent interpolation results for unobserved intermediate domains.

# Training Environment
## Dependencies
python 3.6.5

cv2==3.4.3  
logging==0.5.1.2  
matplotlib==3.1.2  
numpy==1.18.1  
pandas==0.25.3  
PIL==6.0.0  
torch==1.0.1  
torchvision==0.2.2  
torchfile==0.1.0  
tensorboardX==2.1  
protobuf=3.13.0.1  

## anaconda virtual environment
We recommand to use anaconda virtual environment :
```
conda create -n venv python=3.6.5
conda activate venv
conda install pytorch==1.0.1 torchvision==0.2.2 cudatoolkit=10.0 -c pytorch
```
proper version of cudatoolkit is needed.

# commands

to train the encoder, run:
```
python3 train_encoder.py
```

to train the translator, run:
```
python3 train_translator.py
```
