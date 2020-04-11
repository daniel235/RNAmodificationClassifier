import numpy as np
import pandas as pd
import tkinter, tkinter.filedialog
import os
import platform
import h5py


'''
    extract psuedouridine locations and use dataframe for loss function
    extract signal for input data
'''

def get_file(read_mode=None, dev=None):
    #in development
    if dev == 1:
        #control
        file = open("./data/control_hela_PSU_signals_v5[7219].txt", 'r')
        return file
    elif dev == 2:
        #modified
        file = open("./data/post_epi_hela_PSU_signals_1mer[7207].txt")
        return file

    if platform.system() == 'Windows':
        #select file
        root = tkinter.Tk()
        if read_mode == None:
            file = tkinter.filedialog.askopenfile(parent=root, mode='rb', title='Choose a file')
        else:
            file = tkinter.filedialog.askopenfile(parent=root, mode=read_mode, title='Choose a file')

        if file != None:
            return file
    
    else:
        #manualy select file
        #list files
        list_of_files = os.listdir("./data")
        print("List of files in data directory")
        for i in range(len(list_of_files)):
            print((i + 1), ' ', list_of_files[i])

        choice = input("Select File ")
        if choice != None:
            fname = "./data/" + list_of_files[int(choice) - 1]
            return open(fname, 'r')

    #if no file selected end run
    raise Exception


def extract_modification():
    #get file
    bed_file = get_file()
    #import file into a data frame
    df = pd.read_csv(bed_file, sep='\t', header=None)
    #set dataframe column names
    header = ['chrom', 'chromStart', 'chromEnd', 'name', 'temp', 'strand']
    df.columns = header[:len(df.columns)]
    print(df)
    bed_file.close()
    return df
    

def signal_reader(filename=None):
    #select file
    if filename != None:
        #hdf = h5py.File(filename, 'r+')
        '''with get_fast5_file(filename, mode="r") as f5:
            print(f5.)
            for read in f5.get_reads():
                data = read.get_raw_data()
                print(data)
        '''
    #using text file
    sigFile = get_file()
    df = pd.read_csv(sigFile, sep='\t', header=None)
    sigFile.close()
    print(df)
    return df

def get_Hela():
    helaFile = get_file(dev=1)
    dfControl = pd.read_csv(helaFile, sep=' ', header=None)
    print(dfControl)
    helaFile = get_file(dev=2)
    dfModified = pd.read_csv(helaFile, sep=' ', header=None)
    print(dfModified)
    return dfControl, dfModified

def get_control_Hela():
    controlFile = get_file()
    dfControl = pd.read_csv(controlFile, sep=' ', header=None)
    return dfControl

def get_pseudo_Hela():
    pseudoFile = get_file()
    dfModified = pd.read_csv(pseudoFile, sep=' ', header=None)
    return dfModified

def event_reader(filename=None):
    eventFile = get_file()
    df = pd.read_csv(eventFile, sep="\t", header=None, index_col=None)
    eventFile.close()
    #create headers
    header = ['mean', 'stdv', 'start', 'length', 'model_state', 'move', 'weights', 'p_model_state', 'mp_state', 'p_mp_state', 'p_A', 'p_C', 'p_G', 'p_T']
    df.columns = header[:len(df.columns)]
    return df

#extract_modification()
#event_reader("./read_746_signals.txt")
#get_Hela()

