# Clinically Informed Imputation of Medical Data Using Parallel Diffusion Models  

This is the **official implementation** of the paper:  

> **Clinically Informed Imputation of Medical Data Using Parallel Diffusion Models**  
> by *[Shuaixun Wang, Xueer Zhang, Sharon Jewell, and Martyn Boutelle, 2025]*  

---

## 📌 Overview  
This repository contains the implementation of the **Parallel Imputation Diffusion Model (PIDM)** proposed in the paper. PIDM is designed for imputing missing values in medical time series, particularly under **high missing rates** and **long sequences**, while ensuring **clinical fidelity**.  

Key features:  
- **Parallel diffusion framework** for long-sequence imputation.  
- **Dynamic Programming Alignment (DPA)** to ensure smooth reconstruction of full sequences.  
- **Clinically Relevance Curve (CRC)** evaluation to assess physiological plausibility beyond standard statistical metrics.  
- Validated on intracranial pressure (ICP) recordings from **CHARIS** and **KCH** datasets.  

---

## 🔧 Installation  

Clone the repo:  
```bash
git clone https://github.com/your-username/PIDM.git
cd PIDM
```

Install dependencies:  
```bash
pip install -r requirements.txt
```

---

## 🚀 Usage  

The whole codes are provided in PIDM.ipynb as jupyter notebook.

## 📊 Results  
PIDM achieves:  
- Superior **DTW alignment** compared to baseline models.  
- Clinically faithful reconstructions validated with **AMP** and **HFC** indices.  
- Flexible adaptation to variable missing lengths (see Appendix I of the paper).  

Please refer to the paper for detailed results.  

---

## 📂 Repository Structure  
```
├── PIDM.ipynb         # Jupyter notebooks containing all codes
├── utils.py           # Functions used in data processing
├── util.py            # Functions used during training the model
├── signal_processing_functions.py # Functions used for signal processing, including a low pass filter
├── requirements.txt   # Python dependencies
└── README.md          # This file
```

---

## 📜 Citation  
If you find this code useful, please cite our work:  

```bibtex
@article{YourName2025PIDM,
  title={Clinically Informed Imputation of Medical Data Using Parallel Diffusion Models},
  author={Shuaixun Wang, Xueer Zhang, Sharon Jewell, and Martyn Boutelle},
  year={2025}
}
```
