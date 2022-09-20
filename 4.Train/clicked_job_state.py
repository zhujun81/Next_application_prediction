import numpy as np
from copy import deepcopy
from collections import Counter
from itertools import permutations
from scipy.sparse import csr_matrix,SparseEfficiencyWarning
import warnings
warnings.simplefilter('ignore',SparseEfficiencyWarning)


class ClickedJobsState:
	def __init__(self, recent_clicks_buffer_max_size, recent_clicks_for_normalization, num_jobs):
		self.recent_clicks_buffer_max_size = recent_clicks_buffer_max_size
		self.recent_clicks_for_normalization = recent_clicks_for_normalization
		self.num_jobs = num_jobs      
		self.reset_state()


	def reset_state(self):
		#Global state

		self.jobs_pop = np.zeros(shape=[self.num_jobs], dtype=np.int64)    
		self.jobs_recent_pop = np.zeros(shape=[self.num_jobs], dtype=np.int64)

		self._update_recent_pop_norm(self.jobs_recent_pop)

		#Clicked buffer has two columns (job_id, click_timestamp)
		self.pop_recent_clicks_buffer = np.zeros(shape=[self.recent_clicks_buffer_max_size, 2], dtype=np.int64)
		self.pop_recent_buffer_job_id_column = 0
		self.pop_recent_buffer_timestamp_column = 1


		#State shared by JobCooccurrenceRecommender and JobKNNRecommender
		self.jobs_coocurrences = csr_matrix((self.num_jobs, self.num_jobs), dtype=np.int64)


		#States specific for balines
		self.baselines_states = dict()

		#Stores the timestamp of the first click in the job
		self.jobs_first_click_ts = dict()
		#Stores the delay (in minutes) from job's first click to job's first recommendation from CHAMELEON
		self.jobs_delay_for_first_recommendation = dict()

		self.current_step = 0
		self.jobs_first_click_step = dict()



	def _update_recent_pop_norm(self, jobs_recent_pop):
		#Minimum value for norm_pop, to avoid 0
		min_norm_pop = 1.0/self.recent_clicks_for_normalization
		self.jobs_recent_pop_norm = np.maximum(jobs_recent_pop / (jobs_recent_pop.sum() + 1), [min_norm_pop])


	def save_state_checkpoint(self):
		self.jobs_pop_chkp = np.copy(self.jobs_pop)
		self.pop_recent_clicks_buffer_chkp = np.copy(self.pop_recent_clicks_buffer)
		self.jobs_coocurrences_chkp = csr_matrix.copy(self.jobs_coocurrences)
		self.baselines_states_chkp = deepcopy(self.baselines_states) 
		self.jobs_first_click_ts_chkp = deepcopy(self.jobs_first_click_ts)
		self.jobs_delay_for_first_recommendation_chkp = deepcopy(self.jobs_delay_for_first_recommendation)  


	def restore_state_checkpoint(self):
		self.jobs_pop = self.jobs_pop_chkp
		del self.jobs_pop_chkp
		self.pop_recent_clicks_buffer = self.pop_recent_clicks_buffer_chkp
		del self.pop_recent_clicks_buffer_chkp
		self.jobs_coocurrences = self.jobs_coocurrences_chkp
		del self.jobs_coocurrences_chkp
		self.jobs_first_click_ts = self.jobs_first_click_ts_chkp
		del self.jobs_first_click_ts_chkp
		self.jobs_delay_for_first_recommendation_chkp = self.jobs_delay_for_first_recommendation_chkp
		del self.jobs_delay_for_first_recommendation_chkp
		self.baselines_states = self.baselines_states_chkp
		del self.baselines_states_chkp


	def get_jobs_pop(self):
		return self.jobs_pop

	def get_jobs_recent_pop_norm(self):
		return self.jobs_recent_pop_norm

	def get_recent_clicks_buffer(self):
		#Returns only the first column (article_id)
		return self.pop_recent_clicks_buffer[:,self.pop_recent_buffer_job_id_column]

	def get_jobs_coocurrences(self):
		return self.jobs_coocurrences

	def update_jobs_coocurrences(self, batch_clicked_jobs):
		for session_jobs in batch_clicked_jobs:
			session_pairs = permutations(session_jobs[np.nonzero(session_jobs)], r=2)
			rows, cols = zip(*session_pairs)
			self.jobs_coocurrences[rows, cols] += 1


	def _update_recently_clicked_jobs_buffer(self, batch_clicked_jobs, batch_clicked_timestamps):
		#Concatenating column vectors of batch clicked jobs
		batch_recent_clicks_timestamps = np.hstack([batch_clicked_jobs.reshape(-1,1), batch_clicked_timestamps.reshape(-1,1)])
		#Inverting the order of clicks, so that latter clicks are now the first in the vector
		batch_recent_clicks_timestamps = batch_recent_clicks_timestamps[::-1]

		#Keeping in the buffer only clicks within the last N hours
		min_timestamp_batch = np.min(batch_clicked_timestamps)

		self.truncate_last_hours_recent_clicks_buffer(min_timestamp_batch)
		#Concatenating batch clicks with recent buffer clicks, limited by the buffer size
		self.pop_recent_clicks_buffer = np.vstack([batch_recent_clicks_timestamps, self.pop_recent_clicks_buffer])[:self.recent_clicks_buffer_max_size]
		#Complete buffer with zeroes if necessary
		if self.pop_recent_clicks_buffer.shape[0] < self.recent_clicks_buffer_max_size:
			self.pop_recent_clicks_buffer = np.vstack([self.pop_recent_clicks_buffer, np.zeros(shape=[self.recent_clicks_buffer_max_size-self.pop_recent_clicks_buffer.shape[0], 2], dtype=np.int64)])


	def truncate_last_hours_recent_clicks_buffer(self, reference_timestamp):
		MILISECS_BY_HOUR = 1000 * 60 * 60     
		min_timestamp_buffer_threshold = reference_timestamp - int(1 * MILISECS_BY_HOUR)
		self.pop_recent_clicks_buffer = self.pop_recent_clicks_buffer[self.pop_recent_clicks_buffer[:,self.pop_recent_buffer_timestamp_column]>=min_timestamp_buffer_threshold]


	def _update_recent_pop_norm(self, jobs_recent_pop):
		#Minimum value for norm_pop, to avoid 0
		min_norm_pop = 1.0/self.recent_clicks_for_normalization
		self.jobs_recent_pop_norm = np.maximum(jobs_recent_pop / (jobs_recent_pop.sum() + 1), [min_norm_pop])


	def _update_recent_pop_jobs(self):
		#Using all the buffer to compute jobs popularity
		pop_recent_clicks_buffer_jobs = self.pop_recent_clicks_buffer[:, self.pop_recent_buffer_job_id_column]
		recent_clicks_buffer_nonzero = pop_recent_clicks_buffer_jobs[np.nonzero(pop_recent_clicks_buffer_jobs)]
		recent_clicks_job_counter = Counter(recent_clicks_buffer_nonzero)

		self.jobs_recent_pop = np.zeros(shape=[self.num_jobs], dtype=np.int64)
		self.jobs_recent_pop[list(recent_clicks_job_counter.keys())] = list(recent_clicks_job_counter.values())

		self._update_recent_pop_norm(self.jobs_recent_pop)

	def _update_pop_jobs(self, batch_jobs_nonzero):
		batch_job_counter = Counter(batch_jobs_nonzero)
		self.jobs_pop[list(batch_job_counter.keys())] += list(batch_job_counter.values())


	def update_jobs_state(self, batch_clicked_jobs, batch_clicked_timestamps):
		self._update_recently_clicked_jobs_buffer(batch_clicked_jobs, batch_clicked_timestamps)
		self._update_recent_pop_jobs()

		self._update_pop_jobs(batch_clicked_jobs)   
