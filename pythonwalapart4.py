from __future__ import division
import nltk, webbrowser, os
from nltk.tag.util import untag
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tree import Tree
from bs4 import BeautifulSoup
#from rake_nltk import Rake
from PIL import Image
import re, warnings, urllib2, os, cookielib, json, nltk, sys, math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import caption_generator
from keras.preprocessing import sequence
import cPickle as pickle
from keras.preprocessing import image
from vgg16 import VGG16
import numpy as np 
from keras.applications.imagenet_utils import preprocess_input
from nltk.corpus import wordnet as wn
from nltk.corpus import brown
###################### Sentence check part ##############################

ALPHA = 0.2
BETA = 0.45
ETA = 0.4
PHI = 0.2
DELTA = 0.85

brown_freqs = dict()
N = 0
name = str(raw_input())

def get_best_synset_pair(word_1, word_2):
    max_sim = -1.0
    synsets_1 = wn.synsets(word_1)
    synsets_2 = wn.synsets(word_2)
    if len(synsets_1) == 0 or len(synsets_2) == 0:
        return None, None
    else:
        max_sim = -1.0
        best_pair = None, None
        for synset_1 in synsets_1:
            for synset_2 in synsets_2:
               sim = wn.path_similarity(synset_1, synset_2)
               if sim > max_sim:
                   max_sim = sim
                   best_pair = synset_1, synset_2
        return best_pair

def length_dist(synset_1, synset_2):
    l_dist = sys.maxint
    if synset_1 is None or synset_2 is None: 
        return 0.0
    if synset_1 == synset_2:
        # if synset_1 and synset_2 are the same synset return 0
        l_dist = 0.0
    else:
        wset_1 = set([str(x.name()) for x in synset_1.lemmas()])        
        wset_2 = set([str(x.name()) for x in synset_2.lemmas()])
        if len(wset_1.intersection(wset_2)) > 0:
            # if synset_1 != synset_2 but there is word overlap, return 1.0
            l_dist = 1.0
        else:
            # just compute the shortest path between the two
            l_dist = synset_1.shortest_path_distance(synset_2)
            if l_dist is None:
                l_dist = 0.0
    # normalize path length to the range [0,1]
    return math.exp(-ALPHA * l_dist)

def hierarchy_dist(synset_1, synset_2):
    h_dist = sys.maxint
    if synset_1 is None or synset_2 is None: 
        return h_dist
    if synset_1 == synset_2:
        # return the depth of one of synset_1 or synset_2
        h_dist = max([x[1] for x in synset_1.hypernym_distances()])
    else:
        # find the max depth of least common subsumer
        hypernyms_1 = {x[0]:x[1] for x in synset_1.hypernym_distances()}
        hypernyms_2 = {x[0]:x[1] for x in synset_2.hypernym_distances()}
        lcs_candidates = set(hypernyms_1.keys()).intersection(
            set(hypernyms_2.keys()))
        if len(lcs_candidates) > 0:
            lcs_dists = []
            for lcs_candidate in lcs_candidates:
                lcs_d1 = 0
                if hypernyms_1.has_key(lcs_candidate):
                    lcs_d1 = hypernyms_1[lcs_candidate]
                lcs_d2 = 0
                if hypernyms_2.has_key(lcs_candidate):
                    lcs_d2 = hypernyms_2[lcs_candidate]
                lcs_dists.append(max([lcs_d1, lcs_d2]))
            h_dist = max(lcs_dists)
        else:
            h_dist = 0
    return ((math.exp(BETA * h_dist) - math.exp(-BETA * h_dist)) / 
        (math.exp(BETA * h_dist) + math.exp(-BETA * h_dist)))
    
def word_similarity(word_1, word_2):
    synset_pair = get_best_synset_pair(word_1, word_2)
    return (length_dist(synset_pair[0], synset_pair[1]) * 
        hierarchy_dist(synset_pair[0], synset_pair[1]))

def most_similar_word(word, word_set):
    max_sim = -1.0
    sim_word = ""
    for ref_word in word_set:
      sim = word_similarity(word, ref_word)
      if sim > max_sim:
          max_sim = sim
          sim_word = ref_word
    return sim_word, max_sim
    
def info_content(lookup_word):
    global N
    if N == 0:
        # poor man's lazy evaluation
        for sent in brown.sents():
            for word in sent:
                word = word.lower()
                if not brown_freqs.has_key(word):
                    brown_freqs[word] = 0
                brown_freqs[word] = brown_freqs[word] + 1
                N = N + 1
    lookup_word = lookup_word.lower()
    n = 0 if not brown_freqs.has_key(lookup_word) else brown_freqs[lookup_word]
    return 1.0 - (math.log(n + 1) / math.log(N + 1))
    
def semantic_vector(words, joint_words, info_content_norm):
    sent_set = set(words)
    semvec = np.zeros(len(joint_words))
    i = 0
    for joint_word in joint_words:
        if joint_word in sent_set:
            # if word in union exists in the sentence, s(i) = 1 (unnormalized)
            semvec[i] = 1.0
            if info_content_norm:
                semvec[i] = semvec[i] * math.pow(info_content(joint_word), 2)
        else:
            # find the most similar word in the joint set and set the sim value
            sim_word, max_sim = most_similar_word(joint_word, sent_set)
            semvec[i] = PHI if max_sim > PHI else 0.0
            if info_content_norm:
                semvec[i] = semvec[i] * info_content(joint_word) * info_content(sim_word)
        i = i + 1
    return semvec                
            
def semantic_similarity(sentence_1, sentence_2, info_content_norm):

    words_1 = nltk.word_tokenize(sentence_1)
    words_2 = nltk.word_tokenize(sentence_2)
    joint_words = set(words_1).union(set(words_2))
    vec_1 = semantic_vector(words_1, joint_words, info_content_norm)
    vec_2 = semantic_vector(words_2, joint_words, info_content_norm)
    return np.dot(vec_1, vec_2.T) / (np.linalg.norm(vec_1) * np.linalg.norm(vec_2))

def word_order_vector(words, joint_words, windex):

    wovec = np.zeros(len(joint_words))
    i = 0
    wordset = set(words)
    for joint_word in joint_words:
        if joint_word in wordset:
            # word in joint_words found in sentence, just populate the index
            wovec[i] = windex[joint_word]
        else:
            # word not in joint_words, find most similar word and populate
            # word_vector with the thresholded similarity
            sim_word, max_sim = most_similar_word(joint_word, wordset)
            if max_sim > ETA:
                wovec[i] = windex[sim_word]
            else:
                wovec[i] = 0
        i = i + 1
    return wovec

def word_order_similarity(sentence_1, sentence_2):
    words_1 = nltk.word_tokenize(sentence_1)
    words_2 = nltk.word_tokenize(sentence_2)
    joint_words = list(set(words_1).union(set(words_2)))
    windex = {x[1]: x[0] for x in enumerate(joint_words)}
    r1 = word_order_vector(words_1, joint_words, windex)
    r2 = word_order_vector(words_2, joint_words, windex)
    return 1.0 - (np.linalg.norm(r1 - r2) / np.linalg.norm(r1 + r2))

def similarity(sentence_1, sentence_2, info_content_norm):
    return DELTA * semantic_similarity(sentence_1, sentence_2, info_content_norm) + \
        (1.0 - DELTA) * word_order_similarity(sentence_1, sentence_2)

##########################Captioning part#################################
cg = caption_generator.CaptionGenerator()
def load_image(path):
    img = image.load_img(path, target_size=(224,224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return np.asarray(x)
def process_caption(caption):
	caption_split = caption.split()
	processed_caption = caption_split[1:]
	try:
		end_index = processed_caption.index('<end>')
		processed_caption = processed_caption[:end_index]
	except:
		pass
	return " ".join([word for word in processed_caption])

def get_encoding(model, img):
	image = load_image('Pictures/' + name.split('.')[0] +'/' +str(img))
	pred = model.predict(image)
	pred = np.reshape(pred, pred.shape[1])
	return pred

def get_best_caption(captions):
    captions.sort(key = lambda l:l[1])
    best_caption = captions[-1][0]
    return " ".join([cg.index_word[index] for index in best_caption])

def get_all_captions(captions):
    final_captions = []
    captions.sort(key = lambda l:l[1])
    for caption in captions:
        text_caption = " ".join([cg.index_word[index] for index in caption[0]])
        final_captions.append([text_caption, caption[1]])
    return final_captions
    
def load_encoding_model():
	model = VGG16(weights='imagenet', include_top=True, input_shape = (224, 224, 3))
	return model
	
def generate_captions(model, image, beam_size):
	start = [cg.word_index['<start>']]
	captions = [[start,0.0]]
	while(len(captions[0][0]) < cg.max_cap_len):
		temp_captions = []
		for caption in captions:
			partial_caption = sequence.pad_sequences([caption[0]], maxlen=cg.max_cap_len, padding='post')
			next_words_pred = model.predict([np.asarray([image]), np.asarray(partial_caption)])[0]
			next_words = np.argsort(next_words_pred)[-beam_size:]
			for word in next_words:
				new_partial_caption, new_partial_caption_prob = caption[0][:], caption[1]
				new_partial_caption.append(word)
				new_partial_caption_prob+=next_words_pred[word]
				temp_captions.append([new_partial_caption,new_partial_caption_prob])
		captions = temp_captions
		captions.sort(key = lambda l:l[1])
		captions = captions[-beam_size:]

	return captions

def test_model(weight, img_name, beam_size = 3):
	encoded_images = pickle.load( open( "encoded_images.p", "rb" ) )
	model = cg.create_model(ret_model = True)
	model.load_weights(weight)
	#print '\n\n\n',img_name,'\n\n\n\n\n\n'
	image = encoded_images[img_name]
	captions = generate_captions(model, image, beam_size)
	return process_caption(get_best_caption(captions))
	#return [process_caption(caption[0]) for caption in get_all_captions(captions)] 
	
def test_model_on_images(weight, img_dir, beam_size = 3):
	imgs = []
	captions = {}
	with open(img_dir, 'rb') as f_images:
		imgs = f_images.read().strip().split('\n')
	print 'imgs in test model:     ', imgs
	encoded_images = pickle.load( open( "encoded_images.p", "rb" ) )
	model = cg.create_model(ret_model = True)
	model.load_weights(weight)
	#print '\n\n\n\n',imgs,'\n\n\n\n\n'
	f_pred_caption = open('predicted_captions.txt', 'wb')

	for count, img_name in enumerate(imgs):
		print "Predicting for image: "+str(count)
		image = encoded_images[img_name]
		image_captions = generate_captions(model, image, beam_size)
		best_caption = process_caption(get_best_caption(image_captions))
		captions[img_name] = best_caption
		print img_name+" : "+str(best_caption)
		f_pred_caption.write(img_name+"\t"+str(best_caption))
		f_pred_caption.flush()
	f_pred_caption.close()

############################ Scraping part ######################################
def get_soup(url,header):
    return BeautifulSoup(urllib2.urlopen(urllib2.Request(url,headers=header)),'html.parser')

def function(words,i,DirName):
	query = words# you can change the query for the image  here
	image_type="ActiOn"
	url="https://www.google.co.in/search?q="+query+"&source=lnms&tbm=isch"
#	print url
	#add the directory for your image here
	DIR="Pictures" 
	header={'User-Agent':"Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/43.0.2357.134 Safari/537.36"
	}
	soup = get_soup(url,header)
	ActualImages=[]# contains the link for Large original images, type of  image
	cnt = 0
	for a in soup.find_all("div",{"class":"rg_meta"}):
	    link , Type =json.loads(a.text)["ou"]  ,json.loads(a.text)["ity"]
	    if (cnt < 10):
	        ActualImages.append((link,Type))
	        cnt += 1

	print  "there are total" , len(ActualImages),"images"
	print('DIR: ' + str(DIR))
	if not os.path.exists(DIR):
	            os.mkdir(DIR)
	DIR = os.path.join(DIR, DirName)
	print 'Dir: ',str(DIR) 
	if not os.path.exists(DIR):
		os.mkdir(DIR)
	for i , (img , Type) in enumerate( ActualImages):
	    	try:
	        	req = urllib2.Request(img, headers={'User-Agent' : header})
	        	raw_img = urllib2.urlopen(req).read()
		        cntr = len([i for i in os.listdir(DIR) if image_type in i]) + 1
		        print cntr
		        if len(Type)==0:
		            f = open(os.path.join(DIR , image_type + "_"+ str(cntr)+"." +"jpg"), 'wb')
		       	else :
		       	    print(img)
		            f = open(os.path.join(DIR , image_type + "_"+ str(cntr)+"."+Type), 'wb')
		        f.write(raw_img)
		        f.close()
		except Exception as e:
		        print "could not load : "+img
		        print e
def work1(tokenized,val):
	one = word_tokenize(tokenized)
#		tagged = nltk.pos_tag(one)
#		print(tagged) 
#		chunkgram = r"""Chunk: {<NN|NNS|NNP|NNPS|VB|VBD|VBG|VBN|VBP|VBZ>}"""
#		chunkParser = nltk.RegexpParser(chunkgram)
#		chunked = chunkParser.parse(tagged)
#		print(tokenized[i])
#		chunked.draw()
	stopWords = set(stopwords.words('english'))
	chunked = []
	for i in one:
		if i not in stopWords:
			chunked.append(i)
	flag = 1
	Dir_Name = val.split('.')
	extract = chunked
		#for t in chunked.subtrees(filter = lambda x: x.label() == "Chunk"):
		#	temp = t.leaves()
		#	extract = extract + untag(temp)
#		extract.show()
#		rake_object = Rake()
#		rake_object.extract_keywords_from_text(tokenized[i])
#		extract =  rake_object.get_ranked_phrases()[:3]
#		print extract
	temp1 = [x for x in val.split('.')]
#		print 'temp',temp1 
	temp = [x for x in temp1[0].split('_')]
#		print temp[0], temp[1]
#		print extract
	if len(temp) > 1:
		if (temp[0] not in extract) and (temp[1] not in extract):
			words = "\"" + temp[1] + "\"" + '+' + "\"" + extract[0] + "\""
		else :
			words = extract[0]
	else:
		words = temp[0]
	for j in extract[1:]:
		words = words + '+' + "\"" + j + "\""
	words = words
	print words
	function(words,i,str(Dir_Name[0]))
################################### main ############################
fo = open(name,"rb")
warnings.filterwarnings("ignore")
temp = fo.read().decode('ascii','ignore')
fo.close()
	#print name.split('.')[0]
	#fo.write(temp+'\n')				#for not scraping and writing
tokenized = sent_tokenize(temp)
#for alpha in tokenized:				#for not scraping and writing
#	print alpha
#	with open(name.split('.')[0]+'_captions.txt',"a") as fo:
#		fo.write(alpha + '\n')
#	work1(alpha,name)			#for not scraping and writing
#fo.close()						#for not scraping and writing'''
la = []
weight = 'weights-improvement-120.hdf5'
encoding_model = load_encoding_model()
encoded_images = {}
i = 0
save = ''
fo = open('Test_labels.txt','w')
for f in os.listdir('Pictures/'+ name.split('.')[0]):
	fo.write(f+'\n')
for f in os.listdir('Pictures/'+ name.split('.')[0]):
	print f, f.split('.')[1]
	if 'svg' != f.split('.')[1]:
		encoded_images[f] = get_encoding(encoding_model, f)
		i+=1
		save = f
		#print encoded_images
	else:
		os.remove('Pictures/'+ name.split('.')[0] + '/'+f)
with open( "encoded_images.p", "wb" ) as pickle_f:
		pickle.dump(encoded_images, pickle_f)
la = []
i = 0
for f in os.listdir('Pictures/'+name.split('.')[0]):
	print i
	la.append(test_model(weight,f,beam_size=3))
	i+=1
fo = open(name.split('.')[0]+'_captions.txt',"rU")
imgpicks = []
for line in fo:
	max_val = 0
	loc = 0
	#print line
	for j in range(0,len(la)):
		val = similarity(line,la[j],True)
		if val > max_val:
			max_val = val
			loc = j
	#	print j
	imgpicks.append((loc,max_val))
print imgpicks
fo.close()
############## Display Image #########################
fo = open('Test_labels.txt','r')
for i in range(0,len(imgpicks)):
	val = imgpicks[i][0]
	print val
	temp = []
	temp = fo.read().split('\n')
#	print val
	imgs = plt.imshow(mpimg.imread('Pictures/' + name.split('.')[0] + '/' + temp[val]))
	plt.show()		
