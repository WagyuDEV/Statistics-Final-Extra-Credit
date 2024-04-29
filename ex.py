import datasets as ds
import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.stats as st
# import matplotlib as mpl

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
        # posMean = posMean + len(s['text'])
        posLens.append(len(s['text']))
    elif s['label'] == 0:
        neg.append(s['text'])
        # negMean = negMean + len(s['text'])
        negLens.append(len(s['text']))

print("posTotal", posMean)
print("negTotal", negMean)

# posMean = float(posMean)/len(pos)
# negMean = float(negMean)/len(neg)
posMean = np.mean(posLens)
negMean = np.mean(negLens)

posSD = np.std(posLens)
negSD = np.std(negLens)

# for s in pos:
#     posSD = posSD + math.pow(float(len(s)) - float(posMean), 2.0)
# posSD = math.sqrt(posSD/(len(pos)-1.0))

# for s in neg:
#     negSD = negSD + math.pow(float(len(s)) - float(negMean), 2.0)
# negSD = math.sqrt(negSD/(len(neg)-1.0))

print()
print("pos[0]", pos[0])
print()
print("len(pos)", len(pos))
print("len(pos[0])", len(pos[0]))
print()
print("neg[0]", neg[0])
print()
print("len(neg)", len(neg))
print("len(neg[0])", len(neg[0]))
print()
print("posMean", posMean)
print("negMean", negMean)
print("posSD", posSD)
print("negSD", negSD)

ttest = st.ttest_ind_from_stats(posMean, posSD, len(pos), negMean, negSD, len(neg), equal_var=True, alternative="two-sided")
ttest2 = st.ttest_ind(posLens,negLens)
print("statistic", ttest.statistic)
print("p-value", ttest.pvalue)
print(ttest2)
print(ttest2.confidence_interval(0.95))
print("critical value", st.t.ppf(q=1-.05/2, df=24998))

print()

if ttest.pvalue < 0.05:
    print("Reject the null hypothesis", ttest.pvalue, "< 0.05.", "t-stat =", ttest.statistic,
           "The mean character count between positive and negative reviews is not the same")
else:
    print("Accept the null hypothesis", ttest.pvalue, "> 0.05", "t-stat =", ttest.statistic,
          "The mean character count between positive and negative reviews is the same")

# posmu, posstd = st.norm.fit(posLens)

p1 = plt.figure("Positive")
plt.hist(posLens, bins=[0, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000, 6500, 7000, 7500, 8000], histtype="bar")
# plt.hist(posLens, bins=8, histtype="bar")

# posxmin, posxmax = plt.xlim()
# posx = np.linspace(0, 8000, 100)
# posp = st.norm.pdf(posx, posmu, posstd)

# plt.plot(posx, posp, 'k', linewidth=2)
p1.savefig("positive.png")
# p1.show()

p2 = plt.figure("Negative")
plt.hist(negLens, bins=[0, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000, 6500, 7000, 7500, 8000], histtype="bar")
# plt.hist(negLens, bins=8, histtype="bar")
p2.savefig("negative.png")
# p2.show()

# input()