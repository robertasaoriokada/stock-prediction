import os
import numpy as np
import pandas as pd
import yfinance as yf
from bcb import sgs
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt
import matplotlib.dates as mdates