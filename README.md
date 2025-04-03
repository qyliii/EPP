# EPP: Epitope-Paratope Predictor

## Overview
The binding interface between antigens and antibodies is pivotal in humoral immune responses and provides crucial effective defense against pathogens and exogenous threats. Existing predictive computational methodologies, including structure-based and sequence-based approaches, offer valuable insights but face challenges such as unknown antigen structures and reliance on manually curated features. Most current methods primarily predict antigen epitope, often neglecting the specific molecular epitope-paratope interactions essential for immune efficacy.

In this study, we introduce **EPP (Epitope-Paratope Predictor)**, a novel approach leveraging the **ESM-2 protein language model** as a feature encoder and a **Bi-LSTM network** to predict epitope-paratope interactions. Our method processes antigen and antibody sequences as inputs, leveraging a novel dataset strategy and encoding protein representations to enhance prediction accuracy. The results demonstrate a significant improvement in prediction accuracy compared to existing methods, highlighting the importance of protein feature encoders and temporal dependencies within sequences.


---


## Installation
Clone the repository and install the dependencies:
```bash
git clone https://github.com/qyliii/EPP.git
cd EPP
conda env create -f environment.yml
conda activate epp
```

## Usage
### Training the Model
```bash
python train.py
```

### Running Predictions
```bash
python predict.py --agseq MQIPQAPWPVVW.... --abseq QVQ....
```

## Citation
If you use EPP in your research, please cite:
```
@article{,
  title={Enhanced Prediction of Antigen and Antibody Binding Interface Using ESM-2 and Bi-LSTM},
  author={Qianying Li, Yanmin Zhao, Mahendra D. Chordia, Xiuming Xia, Bo Zhang, Heping Zheng},
  journal={Human Immunology},
  year={2025}
}
```

## License
This project is licensed under the MIT License - see the LICENSE file for details.

