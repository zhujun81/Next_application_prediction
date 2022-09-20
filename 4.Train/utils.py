import os
import math
import json
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf



### File loader ###
def serialize(filename, obj):
	with tf.io.gfile.GFile(filename, 'wb') as handle:
		pickle.dump(obj, handle)

def deserialize(filename):
	with tf.io.gfile.GFile(filename, 'rb') as handle:
		return pickle.load(handle)


### Math helper ###
def log_base(x, base):
	numerator = tf.log(tf.to_float(x))
	denominator = tf.log(tf.constant(base, dtype=numerator.dtype))
	return numerator / denominator


### Data type helper ###
def get_tf_dtype(dtype):
	if dtype == 'int':
		tf_dtype = tf.int64
	elif dtype == 'float':
		tf_dtype = tf.float32
	elif dtype == 'string' or dtype == 'bytes':
		tf_dtype = tf.string
	else:
		raise Exception('Invalid dtype "{}"'.format(dtype))
	return tf_dtype 


def get_embedding_size(unique_val_count, const_mult=8):
	return int(math.floor(const_mult * unique_val_count**0.25))



def merge_two_dicts(x, y):
	#Python 2 to 3.4
	#z = x.copy()   # start with x's keys and values
	#z.update(y)    # modifies z with y's keys and values & returns None
	#return z
	#Python 3.5 or greater
	return {**x, **y}
	

def save_eval_baseline_metrics_csv(eval_sessions_metrics_log, output_dir, output_csv='eval_stats_baselines.csv'):
	metrics_df = pd.DataFrame(eval_sessions_metrics_log)
	metrics_df = metrics_df.reset_index()
	csv_output_path = os.path.join(output_dir, output_csv)
	metrics_df.to_csv(csv_output_path, index=False)


def append_lines_to_text_file(filename, lines):
	with open(filename, "a") as myfile:
		myfile.writelines([line+"\n" for line in lines])


#Saving the negative samples used to evaluate each sessions, so that benchmarks metrics outside the framework (eg. Matrix Factorization) can be comparable
def save_sessions_negative_jobs(model_output_dir, sessions_negative_jobs_list, output_file='eval_sessions_negative_samples.json'):
	append_lines_to_text_file(os.path.join(model_output_dir, output_file), map(lambda x: json.dumps({'session_id': x['session_id'],'negative_jobs': x['negative_jobs']}), sessions_negative_jobs_list))


def save_sessions_model_recommendations_log(model_output_dir, sessions_model_recommendations_log_list, output_file='eval_model_recommendations_log.json'):
	append_lines_to_text_file(os.path.join(model_output_dir, output_file), map(lambda x: json.dumps({
		'session_id': x['session_id'],
		'next_click_labels': x['next_click_labels'],
		'predicted_job_ids': x['predicted_job_ids'],
		'predicted_job_probs': x['predicted_job_probs'],
		'predicted_job_norm_pop': x['predicted_job_norm_pop']}), sessions_model_recommendations_log_list))

