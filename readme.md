# Sleep-Wake Classification using Logistic Regression
## 1. Problem Definition
### Scope and Context
Sleep-wake classification is a critical component of polysomnography (PSG) used to diagnose various sleep disorders. Traditionally, this process involves hours of manual scoring by experts for every single night of data. This project explores the use of Machine Learning to automate this classification, reducing manual labor while maintaining diagnostic accuracy[cite: 90].

### SMART Objective
To develop a supervised machine learning model using a **Logistic Regression** algorithm to classify 30-second epochs of physiological data as either "Sleep" or "Wake" with a target accuracy of >80% by the end of the course.

### Stakeholder Analysis
* **Clinicians & Sleep Technicians:** Primary users who require automated, interpretable tools to accelerate the diagnosis process.
* **Patients:** Stakeholders who benefit from faster results and potentially more accessible sleep health screening.
* **Researchers:** Users who utilize the model's coefficients (interpretability) to understand the physiological features most indicative of wakefulness.

## 2. Data Identification
We utilize the **PhysioNet Sleep-EDF Database (Expanded)**, a gold-standard public dataset.
* **Subject Selection:** 20 Healthy Subjects from the "Sleep Cassette" (SC) study.
* **Night Selection:** Night 2 recordings only. This choice mitigates the "First Night Effect," ensuring the model is trained on stabilized, natural sleep patterns rather than data influenced by the subject's adjustment to the equipment.

## 3. Project Structure
The project is organized to ensure reproducibility and professional standards:
* `data/raw/`: Contains original .edf files fetched from PhysioNet.
* `notebooks/`:
    * `01_data_fetching.ipynb`: Automated script to acquire specific subject nights.
    * `02_eda_preprocessing.ipynb`: Data cleaning and feature extraction (Band Power).
    * `03_model_training.ipynb`: Logistic Regression implementation and evaluation.
* `requirements.txt`: List of dependencies (mne, scikit-learn, pandas, etc.).

## 4. Methodology
As per the course curriculum for Week 2, **Logistic Regression** was chosen for its efficiency in binary classification and its baseline interpretability.

### Preprocessing Workflow
1. **Epoching:** Segmenting continuous EEG signals into 30-second windows[cite: 97].
2. **Feature Extraction:** Calculating relative power in frequency bands (Delta, Theta, Alpha, Sigma, Beta)[cite: 97].
3. **Normalization:** Scaling features using `StandardScaler` to ensure the model converges effectively[cite: 97].
4. **Partitioning:** Splitting data into Training (80%) and Test (20%) sets[cite: 98].

## 5. How to Run
1. Install dependencies: `pip install -r requirements.txt`
2. Run `01_data_fetching.ipynb` to download the dataset.
3. Execute the preprocessing and training notebooks in order.