import os

data_files = [x[2] for x in os.walk('Test')]
data_files[0].sort()
fo = open('Labels_for_test.txt','a')
for i in data_files[0]:
	fo.write(i+'\n')
