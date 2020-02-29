import seaborn as sns
import matplotlib.pyplot as plt
from scipy.fft import fftshift
from scipy import signal
from statistics import mean, stdev


def get_signal_distribution(x, y):
    control_signal_len = []
    pseudo_signal_len = []
    for i in range(len(x)):
        if len(x[i]) < 200:
            if y[i] == 0:
                control_signal_len.append(len(x[i]))
            else:
                pseudo_signal_len.append(len(x[i]))

    #sns.distplot(signal_len, hist=False, kde=True, kde_kws={"shade": True, "linewidth": 3})
    #plt.show()
    sns.kdeplot(control_signal_len, shade=True, label='control')
    sns.kdeplot(pseudo_signal_len, shade=True, label='pseudo')
    plt.legend()
    plt.savefig('signal_distribution.png')
    plt.show()
    plt.close()

    plt.hist(control_signal_len, bins=40, density=True, color='blue')
    plt.hist(pseudo_signal_len, bins=40, density=True, color='green')
    plt.savefig('hist_signal_distribution.png')
    plt.show()
    #plt.gca().set(title='Frequency Histogram', ylabel='Frequency')


def signal_length_score(length, score, filters, kernel):
    #hash length to bin
    mod = int(length / 20)
    with open("./results/binStats.txt", 'a+') as f:
        line = str(score) + " bin " + str(mod) + " " + str(filters) + " " + str(kernel) + "\n"
        f.write(line)

    
def signal_amplitude_mean(x, y):
    control_mean = []
    pseudo_mean = []
    for i in range(len(x)):
        if y[i][0] == 0:
            control_mean.append(mean(x[i]))

        else:
            pseudo_mean.append(mean(x[i]))

    #plot distribution
    sns.kdeplot(control_mean, shade=True, label='control')
    sns.kdeplot(pseudo_mean, shade=True, label='pseudo')
    plt.legend()
    plt.savefig('signal_mean_distribution.png')
    plt.show()
    plt.close()

#get standard deviation of each signal 
def std_deviation_distribution(x, y):
    control_deviation = []
    pseudo_deviation = []
    for i in range(len(x)):
        if y[i][0] == 0:
            control_deviation.append(stdev(x[i]))

        else:
            pseudo_deviation.append(stdev(x[i]))
    

    #plot distribution
    sns.kdeplot(control_deviation, shade=True, label='control')
    sns.kdeplot(pseudo_deviation, shade=True, label='pseudo')
    plt.legend()
    plt.savefig('signal_deviation_distribution.png')
    plt.show()
    plt.close()

        
        