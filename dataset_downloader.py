import urllib
import os
import zipfile

DATASET_FOLDER = 'twitter-sentiment-dataset/'
URL = 'http://thinknook.com/wp-content/uploads/2012/09/Sentiment-Analysis-Dataset.zip'

if not os.path.exists(DATASET_FOLDER):
    os.mkdir(DATASET_FOLDER)
zip_file = ''.join([DATASET_FOLDER, 'sentiment-dataset.csv.zip'])

print 'Downloading dataset...'
retriever = urllib.URLopener()
retriever.retrieve(URL, zip_file)
print "Done."

print 'Extracting data...'
z = zipfile.ZipFile(zip_file)
z.extractall(DATASET_FOLDER)
z.close()
old_file = ''.join([DATASET_FOLDER, 'Sentiment Analysis Dataset.csv'])
new_file = ''.join([DATASET_FOLDER, 'sentiment-dataset.csv'])
os.rename(old_file, new_file)
os.remove(zip_file)
print 'Done. Exiting...'
