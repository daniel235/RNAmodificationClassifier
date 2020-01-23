import pseudoExtractor as ps


#start pseudoExtractor 
controlHela, psuedoHela = ps.get_Hela()

#omit file name
controlHela = controlHela.drop(controlHela.columns[0], axis=1)
pseudoHela = psuedoHela.drop(psuedoHela.columns[0], axis=1)

print(controlHela)

kmerData = {}
for i in range(len(controlHela)):
    print(controlHela.iloc[i])
