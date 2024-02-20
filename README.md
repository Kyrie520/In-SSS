# In-SSS
## Overview
Stochastic computing is crucial for accurate and efficient processing of massive data in advanced artificial intelligence applications. This project demonstrated **a neuromorphic computing simulation based on the In-SSS** using the handwritten artificial neural network. We took a pattern recognition task as an example to illustrate the advantages of stochastic computation over deterministic computation. Furthermore, we conducted a comparison of three hardware stochastic computing strategies, and the findings indicated that **the in-situ stochastic computing strategy based on the In-SSS has the best performance**.
## Requirements
This project requires only a standard computer with enough RAM to support the operations. It has been tested on the following environment:
- Windows 11
- RAM: 32 GB
- CPU: 8 cores (3.30 GHz/core)
- tensorflow 2.6.0
> **Note:** The runtimes below are generated using a computer with the above specs.
## Installation Guide
You can use pip to install the required python packages in PyCharm through the command line interface.
For example, to install the numpy package, you can use:
``pip install numpy``
which will install in about a few minutes.
## Usage
You can just run through:
```
python Stochastic computing_Strategy 1 (this work).py
python Stochastic computing_Strategy 2.py
python Stochastic computing_Strategy 3.py
python Deterministic computing.py
```
> **Note:** The model is quite small, so it will take just several minutes to finish the training procedure (about 1.6 s/epoch).
## License
This project is covered under the **Apache 2.0 License**.
