# Extended Evaluation at $224 \times 224 \times 3$
### ℹ️ Background & Motivation

Here we provide extended evaluations on ***ImageNet (ISLVRC2012)*** and ***Lung & Colon Cancer (LC25000)***. These experiments further helps us to decide the better variant, and we will use the better one in the later experiments.

Here are details of these evaluations:

- *Evaluate times*: $5$.
- *Attack Target*: the averaged gradient from a sampled batch with $16$ datapoints.
- *How to sample*: the first $5$ batches, including:
  - `Shuffle=True`: `.ipynb` end with ***random***.
  - `Shuffle=False`: `.ipynb` end with ***fix***.
- *Way to report*: PSNRs in $5$ times recovery, including the ***max*** value and the ***median*** value of them.

### 🚩 How to run

There are two ways to run with it, including directly running with `.ipynb`, and run with scripts in the parent folder:

```bash
$ pwd
~/dev/FedLeak_transfer_test

$ python run_notebook.py
```

### 🔬 What's more

- We also provide the results in our evaluations, please see `./ImageNet/` and `./LC25000` for more details.

- For the summary results, please refer to [`./statistics.xlsx`](./statistics.xlsx).
