
# IDEARS - Integrated Disease Explanation and Associations Risk Scoring

## Overview

This is the codebase for IDEARs - The Integrated Disease Explanation and Associations Risk Scoring. Its overall architecture is shown below:


<img src="fig2.png"
     alt="Markdown Monster icon"
     style="float: left; margin-right: 10px;" />


The code is designed to represent the following situation for prospective studies, which depicts a participant in UKB attending the centre at baseline and then subsequently having a number of outcomes occur

<img src="fig1.png"
     alt="Markdown Monster icon"
     style="float: left; margin-right: 10px;" />


## How to Run
To ease the configuation, please install Anaconda and set this up in a virtual environment. 

1. Install Anaconda:

https://www.anaconda.com/products/individual


## Codebase Structure

### Overview
Import modules etc.

### Directory Tree and Explanations

This folder shows the implementation of the IDEARs platform.

```
📦ukb_IDEARS-pipeline-poc
 ┣ 
 ┃ ┣ src
 ┃ ┃ ┣ idears
 ┃ ┃ ┃ ┣ 📂 preprocessing        
 ┃ ┃ ┃ ┃  ┣  📜 data_proc.py
 ┃ ┃ ┃ ┃  ┣  📜 idears_backend.py 
 ┃ ┃ ┃ ┃ 📂 models
 ┃ ┃ ┃ ┃  ┣  📜 mlv2.py       
 ┃ ┃ ┃ ┃ 📂 frontend
 ┃ ┃ ┃ ┣ ┣ 📜 app1.py
 ┃ ┣ applications
 ┃ ┃ ┃-AD
 ┃ ┃ ┃-PD
 ┣ 📜config.yaml
 ┣ 📜requirements.txt
 ┣ 📜main.py
 ┣ 📜README.md
 ┣
```


## Individual Models

Note for Parkinson's please go to the following link to see the notebooks used to generate the data in our manuscript

"Machine Learning Analysis of the UK Biobank Reveals IGF-1 and Inflammatory Biomarkers Predict Parkinson’s Disease Risk"

https://github.com/MikeAllwright23/idears_orig/tree/main/notebooks/pd

The data behind the figures is also available at this location

https://github.com/MikeAllwright23/idears_orig/tree/main/data

## Enquiries

Michael Allwright - michael@allwrightanalytics.com, michael.allwright@sydney.edu.au

