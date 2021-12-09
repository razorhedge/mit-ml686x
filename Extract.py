import tarfile
filename = 'resources_netflix.tar.gz'
file = tarfile.open('{}'.format(filename))
file.extractall('../Scripts')
file.close()