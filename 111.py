import os
path = './rte50'
files = os.listdir(path)
for i, file in enumerate(files):
    if file[0]=='q':
        NewName = os.path.join(path, file[2:])
        OldName = os.path.join(path, file)
        os.rename(OldName, NewName)
