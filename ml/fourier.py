from scipy.fft import fft
import matplotlib.pyplot as plt
import numpy as np

#get fourier transform of signal and save images
def get_images(signal, output):
    p_count = 0
    c_count = 0
    for i in range(len(signal)):
        x_axis = np.linspace(0, len(signal[i]), num=len(signal[i]), dtype=int)
        sig = np.array(signal[i])  

        #y = y_value_signal(sig)
        y = np.exp(2j * np.pi * sig / len(sig))
    
        y = fft(y)
        plt.grid()
        plt.plot(x_axis, y)

        #return image
        if output[i][0] == 0:
            c_count += 1
            types = "control"
            fname = "./data/images/" + types + str(c_count) + ".png"
            plt.savefig(fname)
            plt.close()
            
        elif output[i][0] == 1:
            p_count += 1
            types ="pseudo"
            fname = "./data/images/" + types + str(p_count) + ".png"
            plt.savefig(fname)
            plt.close()
        


def y_value_signal(signal):
    #return y value of signal
    y = np.sin(50.0 * 2.0*np.pi*signal) + 0.5*np.sin(80.0 * 2.0*np.pi*signal)
    return y



  
'''
x = np.array([1,2,3,4,5,6,7,8,9,10])
x2 = np.array([1,2,3,4,3,2,1,2,3,4])
y_val = y_value_signal(x2)
print(y_val)
y_val = fft(y_val)
plt.plot(x, y_val)
plt.show()
'''