from scipy.fft import fft
import matplotlib.pyplot as plt
import numpy as np


def read_signal(signal, output):
    #convert signal to nd array
    control_flag = False
    pseudo_flag = False
    count = 10
    second_count = 10
    for i in range(len(signal)):
        if output[i][0] == 0 and control_flag == False:
            count -= 1
            if count <= 0:
                control_flag = True


            sig = np.array(signal[i])
            print(sig)
            y = fft(sig)
            ax = plt.subplot()
            plt.grid()
            ax.plot(signal, y)
            plt.plot(y)
            plt.show()
            plt.close()

        elif output[i][0] == 1 and pseudo_flag == False:
            second_count -= 1
            if second_count <= 0:
                pseudo_flag = True

            sig = np.array(signal[i])
            print(sig)
            y = fft(sig)
            plt.grid()
            plt.plot(y)
            plt.show()
            plt.close()


def y_value_signal(signal):
    #return y value of signal
    y = np.sin(50.0 * 2.0*np.pi*signal) + 0.5*np.sin(80.0 * 2.0*np.pi*signal)
    return y
  
    

    