# Optimized Algorithms for Text Clustering with LLM-Generated Constraints

  ## Dependencies
  - python version: 3.10
  - Dependencies can be installed using requirements.txt

  ## Data sets
  - banking77 (example)
  - clinc
  - clinc_domain
  - tweet
  - go_emotion

 ## **Usage**
Follow the steps below to run the project:

### 1. Text Embeddings with `instructor-large`

Download and use the `instructor-large` model to convert your text into vector embeddings.  
Model URL: [https://huggingface.co/hkunlp/instructor-large](https://huggingface.co/hkunlp/instructor-large)

### 2. Generating Constraints

```
python code/constraint_generate/cl_generate.py
python code/constraint_generate/ml_generate.py
```

The generated constraints will be saved in `constraint_generation/Cls_Result` and `constraint_generation/Mls_Result`, and written into `clustering/constrains` using `clustering/utils/get_constrains`.

### 3. Set Must-Link Confidence 

```
python code/clustering/utils/get_ml_confident.py
```

The generated confidence  will be saved in `utils/ml_confidence.json`.

### 4. Run Algorithm of Text Clustering

```
python code/clustering/cl_ml/LSCK-HC.py
python code/clustering/cl_ml/LSCK.py
```

The final results will be saved in `clustering/cluster_result_output`.

  #### Plot the output

  -  ACC, NMI and ARI calculate the degree of agreement between an algorithm's clustering result and its labels.
  -  Measure the quality of LLM-generated constraints in terms of query time and correctness.
