import numpy as np
#pad data

def signal_data(x, y, timestep=120):
    max = 0
    
    paddedx = []
    paddedy = []
    #get longest sequence in data
    '''
    for i in range(len(x)):
        if len(x[i]) > max and len(x[i]) < 200:
            max = len(x[i])
    '''
    max = 600
    #pad all elements to max length
    for i in range(len(x)):
        if type(x[i]) == 'ndarray':
            x[i] = x[i].tolist()

        if len(x[i]) >= max:
            x = x[:max]

        elif len(x[i]) < max:
            #input raw signal data until signal array full
            for j in range(len(x[i]), max):
                x[i].append(0)

        paddedx.append(x[i])
        paddedy.append(y[i])


    paddedx = np.array(paddedx)
    paddedy = np.array(paddedy)
    #reshape
    paddedx.reshape((paddedx.shape[0], max))
    #shuffle
    idx = np.random.choice(len(paddedx), len(paddedx), replace=False)
    paddedx = paddedx[idx]
    paddedy = paddedy[idx]
    #shuffle again
    idx = np.random.choice(len(paddedx), len(paddedx), replace=False)
    paddedx = paddedx[idx]
    paddedy = paddedy[idx]
    return paddedx, paddedy


