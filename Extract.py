import tarfile
filename = 'resources_mnist.tar.gz'
file = tarfile.open('{}'.format(filename))
file.extractall('../Scripts')
file.close()