py-simple-ml
======================

Simple and PURE implementations of different kind of classic ML algorithms.

Linear algorithms:
-----------------
- Linear Regression
- Logistic Regression

Gradient descent algorithms:
----------------------------
- Classic GD
- Stochastic GD (SGD)
- RMSPROP
- ADAM
- NADAM

It MUST be used only for educational purposes!

## How to use?

Install pytorch and pandas libraries:
```shell
pip install pytorch pandas
```
> pytorch is used only for Iris dataset

Then just run:
```shell
# python3 model.py
SGD score is: 0.9
SGD iter number: 1561
SGD time: 0.051373280002735555 sec
RMSPROP score is: 0.9
RMSPROP iter number: 733
RMSPROP time: 0.05896605100133456 sec
ADAM score is: 0.9
ADAM iter number: 800
ADAM time: 0.07846046800841577 sec
NADAM score is: 0.9
NADAM iter number: 797
NADAM time: 0.09323432800010778 sec
       Method  Score Iterations      Time
0      MY SGD    0.9       1561  0.051373
1  MY RMSPROP    0.9        733  0.058966
2     MY ADAM    0.9        800  0.078460
3    MY NADAM    0.9        797  0.093234
```

See [implementation](https://github.com/ab-kily/py-logistic-regression/blob/master/mylearn/linear_regression.py) for details.
