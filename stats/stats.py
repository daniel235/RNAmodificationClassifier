import seaborn as sns
import matplotlib.pyplot as plt

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
    print("got to density")
    sns.kdeplot(control_signal_len, shade=True, label='control')
    sns.kdeplot(pseudo_signal_len, shade=True, label='pseudo')
    plt.legend()
    plt.savefig('signal_distribution.png')
    plt.show()
    plt.close()

    print("got to hist")
    plt.hist(control_signal_len, bins=40, density=True, color='blue')
    plt.hist(pseudo_signal_len, bins=40, density=True, color='green')
    plt.savefig('hist_signal_distribution.png')
    plt.show()
    #plt.gca().set(title='Frequency Histogram', ylabel='Frequency')