import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import datasets as ds

 
# Generate some data for this 
# demonstration.
# data = np.random.normal(170, 10, 250)

imdb = ds.load_dataset("stanfordnlp/imdb")
dataSize = len(imdb['train'])

pos = []
neg = []
posLens = []
negLens = []

posMean = 0
negMean = 0 

print("type(imdb['train'][0]['label])", type(imdb['train'][0]['label']))

for s in imdb['train']:
    if s['label'] == 1:
        pos.append(s['text'])
        posMean = posMean + len(s['text'])
        posLens.append(len(s['text']))
    elif s['label'] == 0:
        neg.append(s['text'])
        negMean = negMean + len(s['text'])
        negLens.append(len(s['text']))
 
# Fit a normal distribution to
# the data:
# mean and standard deviation
mu, std = norm.fit(posLens) 
 
# Plot the histogram.
plt.hist(posLens, bins=25, density=True, alpha=0.6, color='b')
 
# Plot the PDF.
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mu, std)
 
plt.plot(x, p, 'k', linewidth=2)
title = "Fit Values: {:.2f} and {:.2f}".format(mu, std)
plt.title(title)
 
plt.show()