import numpy as np
#pad data

def signal_data(x, y, timestep=120):
    max = 0
    #get longest sequence in data
    for i in range(len(x)):
        if len(x[i]) > max:
            max = len(x[i])

    #pad all elements to max length
    for i in range(len(x)):
        #input raw signal data until signal array full
        for j in range(len(x[i]), max):
            x[i].append(0)

    x = np.array(x)
    #reshape
    x.reshape((x.shape[0], max))
    print(x)
    return x