# Problem 1: SIRENs

In this problem, you will learn the theory underlying Sine Representation Networks (SIRENs), and implement the model yourself. This exercise will serve as an introduction to both neural fields and computer vision research, as you will implement the model (along with many baselines) from scratch.

## Part 1: Understanding the theory

For this part, start by reading the SIRENs paper:

> [_Implicit Neural Representations with Periodic Activation Functions_](https://arxiv.org/abs/2006.09661) by Sitzmann and Martel et al., NeurIPS 2020

Next, answer all the multiple choice questions in `multiple_choice.yml` to test your comprehension.

## Part 2: Implementing SIRENs (and an MLP baseline)

### Installation

It is recommended that you install the relevant packages with

```bash
conda create -n <insert name> python=3.12
```

and then

```bash
conda activate <insert name> && \
  pip3 install -r requirements.txt
```

Everything should run on CPU

**NOTE** that if you have versioning issues, the versions do not really matter other than for the purposes of testing, you can just `pip install -r requirements-no-version.txt` but expect unit tests that have seeded randomness to fail even if your code is correct.

### Instructions

Using your knowledge from the paper, do the following:

- Implement all missing functions in the `problem_1_*.py` files
  - Implement a SIREN (`problem_1_siren.py`)
  - Implement a bog-standard feed-forward MLP (`problem_1_mlp.py`)
  - Implement derivative-calculating functions (`problem_1_gradient.py`, you will need SIREN and MLP already implemented if you want to unit test this)
  - Implement a flexible MLP/SIREN training loop (`problem_1_train.py`)
- Run the notebook `benchmark.ipynb` to check your functions and visualize the training process

**Testing**. You are provided an optional suite of unit tests in `test_sanity.py` which check that your functions work properly and that your multiple choice YAML is properly formatted. Use them with `pytest .` to help debug your code, but **they will not be used to grade**. You can also look at some example outputs in `should_look_like/`. The unit tests use regression-testing based on a "correct" solution by the TAs and the serialized expected outputs are saved in `test_data/` so do not modfiy that folder. To run all tests except the slowest one, use `pytest . -k "not initialization"`.

### Short-answer questions

Answer these questions in a file called `problem_1.pdf`:

- Why are the MLP reconstructions so much less detailed than those produced by the SIREN?
- The image Laplacian produced by the MLP looks strange... what's happening here and why?

## Part 3: Eeking out better performance!

While the reconstruction from our SIREN looks a lot better than the MLP's, it's still not great (especially compared to the results shown in the paper -- notice those weird blobby artifacts!). In this part, you'll experiment with hyperparameters in the models you've already written to achieve better reconstructions with SIREN.

### Specific deliverables

Include plots and analyses for the following investigations in `problem_1.pdf`:

- Benchmark at least 10 additional SIREN/MLP models by varying multiple hyperparameters (e.g., activation functions, model hyperparameters, training parameters). Plot the results of these benchmarks and include in the writeup.
- Optimize the hyperparameters in the SIREN model to achieve the highest PSNR you can on the `astronaut` image. Include the optimal hyperparameters in your writeup.

Your visualization should look like what is in `should_look_like/` (not in the minutiae, but it should display all of your `problem1_gradients.py` implementations over the course of training along with the image as output by your model based on the coordinates, as well as a PSNR curve comparing different outputs).

**For extra credit:** implement another (nontrivial) model from the literature and add it to your benchmarks! If you do this, please describe the model you implemented in your writeup.

## Submission Instructions

Submit the following files:

- `multiple_choice.yml`
- `problem_1_gradients.py`
- `problem_1_mlp.py`
- `problem_1_siren.py`
- `problem_1_train.py`
- `problem_1.pdf`
