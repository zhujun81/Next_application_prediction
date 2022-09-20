import pickle
import argparse
import numpy as np
import pandas as pd

from nltk import FreqDist
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

def create_args_parser():
	parser = argparse.ArgumentParser()
	parser.add_argument('--input_job_csv_path', default='', help='Input job data path.')
	parser.add_argument('--output_job_content_embeddings_path', default='', help='Output job embedding path.')
	parser.add_argument('--content_type', type=str, default='All', help='Content type of job posting to used. (Title, Description, Requirements, All)')

	return parser


def get_words_freq(tokenized_texts):
	words_freq = FreqDist([word for text in tokenized_texts for word in text])
	return words_freq 

def serialize(filename, obj):
	with tf.io.gfile.GFile(filename, 'wb') as handle:
		pickle.dump(obj, handle)


def export_job_content_embeddings(content_job_embeddings, output_job_content_embeddings_path):
	output_path = output_job_content_embeddings_path
	print('Exporting job embeddings to {}'.format(output_path))
	#to_serialize = (acr_label_encoders, articles_metadata_df, content_article_embeddings)
	to_serialize = content_job_embeddings
	serialize(output_path, to_serialize)


def main():
	parser = create_args_parser()
	args = parser.parse_args()

	path =  "../data/"
	dataset = "cb12/"
	raw_path = path + dataset + "raw/" 
	interim_path = path + dataset + "interim/"
	processed_path = path + dataset + "processed/"


	print('Loading job from file: {}'.format(processed_path + args.input_job_csv_path))
	job_df = pd.read_csv(processed_path + args.input_job_csv_path, header=0, sep='\t')
	print('Job data shape: ', job_df.shape)
	print('Unique JobCity: ', len(job_df.JobCity.unique()))
	print('Unique JobState: ', len(job_df.JobState.unique()))
	print('Unique JobCountry: ', len(job_df.JobCountry.unique()))


	content_type = args.content_type
	print('Load tokenized {} texts...'.format(content_type))
	tokenized_texts = [eval(t) for t in job_df_30[content_type + '_tokenized'].values.tolist()]

	print('Computing word frequencies...')
	# A dictionary 
	words_freq= get_words_freq(tokenized_texts)
	print('Number of vocabulary in {} (raw): {}'.format(content_type, len(words_freq)))

	print('Processing documents...')
	tagged_data = [TaggedDocument(words=w, tags=[i]) for i, w in enumerate(tokenized_texts)]  

	print('Training doc2vec')
	max_epochs = 30
	vec_size = 300
	alpha = 0.025
	model = Doc2Vec(
		vector_size=vec_size,
		alpha=alpha, 
		min_alpha=alpha,   
		window=5,
		negative=5,
		min_count=2,                                     
		max_vocab_size=100000,
		dm = 1,
		dm_mean=1,
		workers=6)

	model.build_vocab(tagged_data)

	for epoch in range(max_epochs):
		print('iteration {0}'.format(epoch))
		model.train(tagged_data, total_examples=model.corpus_count, epochs=1) 
		# decrease the learning rate
		model.alpha -= 0.0002
		# fix the learning rate, no decay
		model.min_alpha = model.alpha

	del tokenized_texts

	print('Concatenating job content embeddings, making sure that they are sorted by the encoded JobID')
	job_content_embeddings = np.vstack([model.docvecs[i-1] for i in job_df['JobID_encoded'].values])    
	embedding_for_padding_job = np.mean(job_content_embeddings, axis=0)
	content_job_embeddings_with_padding = np.vstack([embedding_for_padding_job, job_content_embeddings])
	del job_content_embeddings

	#Checking if content job embedding size correspond to the last JobID
	assert content_job_embeddings_with_padding.shape[0] == job_df['JobID_encoded'].tail(1).values[0]+1

	print('Exporting job content embeddings')
	del job_df
	export_job_content_embeddings(content_job_embeddings_with_padding, args.output_job_content_embeddings_path)

if __name__ == '__main__':
	main()



"""
python3 2.Train_d2v_embedding.py \
    --input_job_csv_path 'jobs_14d_30_consider_user_encoded_tokenized.csv' \
    --output_job_content_embeddings_path '../language_models/pickles/jobs_14d_30_consider_user_All_d2v.pickle'
"""