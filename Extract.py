import tarfile
filename = 'resources_sentiment_analysis.tar.gz'
file = tarfile.open('../{}'.format(filename))
file.extractall('../Scripts')
file.close()