# Data Science Interview Task

You are provided with historical auction data for three artists (`artists.tar.gz`). The goal of this task is to train models on auction data to predict the sale price of lots (pieces of art). In this dataset, the sale price is known as the `hammer_price`.

## Notes:
- Please create a branch and do your work in this repository. When you are finished, submit a Pull Request.
- Good solutions will demonstrate strong technical and analytical proficiency, they will not necessarily be complete.
- Please use a seeded random number generator if applicable. All of your work done in building this task should be repeatable by a third party (me).
- Answers to the following questions may be done either in an iPython notebook or in a separate file (markdown, latex, etc.)
- Please write code in a consistent style (use a python style guide of your choice) and properly document your solutions.
- Please make new commits whenever it makes sense to do so. At a minimum, make a commit for each section with a relevant commit message, and try not to commit broken code. We strongly prefer solutions with many small commits over a few large ones.
- You may use any libraries of your choice.
- Some lots will not have a positive value for `hammer_price`. This is generally because they did not sell at auction. We use these, but they can be filtered from your dataset.
- When you are finished, please submit your work by sharing the repository with all authors listed at the bottom of this README.
- Consider (no need to answer): How do you plan to handle categorical data? Different currencies? Can you assume this dataset is i.i.d. given that there is time-variance? How will you structure your work to take advantage of the non i.i.d. properties of this dataset?

## Before you begin:

* Review our schema documentation ([SCHEMA.md](SCHEMA.md))
* Add a .gitignore (github maintains a Python gitignore [here](https://github.com/github/gitignore/blob/master/Python.gitignore))
* We recommend working in python 3.6.5 or greater.
* Set up a python virtual environment. We recommend using [Pipenv](https://pipenv.readthedocs.io/en/latest/) — you can refer to [SETUP.md](SETUP.md) for using Jupyter Notebook in combination with Pipenv (this is encouraged to make setup easier for you, but neither tool is required).

## Tasks:

### Part 1

We would like to train a model to predict the `hammer_price` of lots at upcoming auctions (some upcoming lots are included in your dataset). Note that this means that you can't use future data to predict the past. You may use as many or as few features as you like except for `buyers_premium` (it's a function of the final sale price. See [SCHEMA.md](SCHEMA.md) for more details). This model should be optimized to minimize the relative error of each lot. Good solutions will have a MAPE below 30%.

   - Did you perform any data cleaning, filtering or transformations to improve the model fit?
   - Why did you choose this model?
   - What loss function did you use? Why did you pick this loss function?
   - Describe how your model is validated.
   - What error metrics did you evaluate your model against?
   - Generate and report a histogram of (`hammer_price - predicted_hammer_price`) / `estimate_low` (normalized model residuals). What are the mean and median values of this distribution?

### Part 2
Please choose 2A or 2B.   

#### 2A

We've been tasked to predict the `hammer_price` of auction lots without estimates (`estimate_low`, `estimate_high`). Assume that at any point in time, we have access to historical estimates of lots that have already sold but not to estimates of future lots we would like to predict. Repeat part 1. Briefly discuss your approach. Good solutions will have comparable accuracy to your work in part 1. Present your work as a runnable python (`.py`) file.

#### 2B

Consider that the realized `hammer_price` for an auction lot is only a single sample from the distribution of what a work could have sold for. Imagine that instead of estimating a sample, we would like to estimate a probability mass function of the `hammer_price` and a cumulative distribution function.

  a. Briefly discuss your approach to this problem. Pick a few of the lots that you generated `hammer_price` predictions for and do this. Do any of the distributions follow a familiar form? How does the variance differ between plots?

  b. Compare your `hammer_price` distributions to the global distribution of all `hammer_price`s. You may consider normalizing them by dividing by one of the estimates. How do they diverge from the global distribution? How confident are you in your results?

## Questions:

Please answer each in a few sentences.

1. Which features are most important for your model? Are there any features that surprised you? Given more data, describe other features you feel would be useful in improving the accuracy of your model.

2. Now, assume we care much more about not over-predicting `hammer_price` than we do about under-predicting the `hammer_price`. Describe how you would go about changing your solution in terms of the model, objective function, etc.

3. Given more time but no new features, how much do you think you could improve the accuracy of your models by? Why? Now assume that we structure all of the information in the observable universe. What types of new features do you expect to have the greatest impact in performance?

4. Was this fun? Which sections / questions were the most difficult and which were the easiest?

### Reviewers
* [Michael D'Angelo (@mldangelo)](http://github.com/mldangelo) ([michael@arthena.com](mailto:michael@arthena.com))
* [Adrian Wisernig (@envoked)](http://github.com/envoked) ([adrian@arthena.com](mailto:adrian@arthena.com))
* [Will Holley (@vivism)](http://github.com/vivism) ([will@arthena.com](mailto:will@arthena.com))
