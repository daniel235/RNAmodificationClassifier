import numpy as np
#pad data

def signal_data(x, y, timestep=120):
    max = 0
    paddedx = []
    paddedy = []
    #get longest sequence in data
    for i in range(len(x)):
        if len(x[i]) > max and len(x[i]) < 200:
            max = len(x[i])

    #pad all elements to max length
    for i in range(len(x)):
        if len(x[i]) < 200:
            #input raw signal data until signal array full
            for j in range(len(x[i]), max):
                x[i].append(0)

            paddedx.append(x[i])
            paddedy.append(y[i])



    paddedx = np.array(paddedx)
    #reshape
    paddedx.reshape((paddedx.shape[0], max))
    return paddedx, paddedy


