import re

"""
Reads the csv dataset available at http://thinknook.com/wp-content/uploads/2012/09/Sentiment-Analysis-Dataset.zip and splits it into two files (.pos and .neg) containing the positive and negative tweets. 
Does some word preprocessing during the parsing.
"""
try: 
	full_dataset = open("twitter-sentiment-dataset/sentiment-dataset.csv", "r")	
	pos_dataset = open("twitter-sentiment-dataset/tw-data.pos", "w")
	neg_dataset = open("twitter-sentiment-dataset/tw-data.neg", "w")
except:
	print "Failed to open file"
	quit()

csv_lines = full_dataset.readlines()
i=0.0

for line in csv_lines:
	i += 1.0
	line = line.split(",", 3)
	tweet = line[3].strip()
	new_tweet = ''

	print "{0:.0f}%".format((i/len(csv_lines)) * 100)

	for word in tweet.split():
		# String preprocessing
		if re.match('^.*@.*', word):
			word = '<NAME/>'
		if re.match('^.*http://.*', word):
			word = '<LINK/>'
		word = word.replace('#', '<HASHTAG/> ')
		word = word.replace('&quot;', ' \" ')
		word = word.replace('&amp;', ' & ')
		word = word.replace('&gt;', ' > ')
		word = word.replace('&lt;', ' < ')
		new_tweet = ' '.join([new_tweet, word])
		
	tweet = new_tweet.strip() + '\n'

	if line[1].strip() == '1':
		pos_dataset.write(tweet)
	else:
		neg_dataset.write(tweet)