# optimizing-matmul-exercises

hi there! this repo is an hands on accompaniment to this great article: https://siboehm.com/articles/22/CUDA-MMM

along with the above article, this notebook will guide you through writing progressively fancier CUDA kernels, with explanations and skeleton code to assist you. 

so far, 6 have been implemented, so you'll end up with 80% of the performance of CUBLASS (Nvidia's official matrix multiplication kernel)

click 'Open in Colab' to run this notebook. you'll be working on a personal copy and any changes you make won't affect the original notebook.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/arb8020/optimizing-matmul-exercises/blob/main/notebooks/optimizing_cuda_matmul.ipynb)


## project structure 

here's what you'll find in this repo:

- `notebooks/`: 
  - `optimizing_cuda_matmul.ipynb`: interactive notebook
- `solutions/`: (try to give each kernel an honest effort first, and let me know where you got stuck) 
  - `kernel_1_solution.cpp` to `kernel_6_solution.cpp`: solutions for each kernel
  - `roofline_solutions.py`: solution for the roofline model exercise
- `src/`: utility stuff
  - `helper_functions.py`: testing functions and such 
  - `test_sgemm.cu`, `test_sgemm2.cu`, `test_sgemm3.cu`: test files for our kernels

## rough prereqs

- basic programming skills (if you can write a for loop in c++/python, you're prob good)
- curiosity about making gpus go zoom 🏎️
