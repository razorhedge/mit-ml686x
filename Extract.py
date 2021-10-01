import tarfile
file = tarfile.open('../resources_sentiment_analysis.tar.gz')
file.extractall('../Scripts')
file.close()