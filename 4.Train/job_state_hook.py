import numpy as np
import tensorflow as tf

from time import time
from metrics import HitRate, MRR, NDCG
from utils import merge_two_dicts, get_tf_dtype, serialize
from evaluation import update_metrics, compute_metrics_results

class JobsStateUpdaterHook(tf.estimator.SessionRunHook):
    """Saves summaries during eval loop."""

    def __init__(self,
                 mode, 
                 model, 
                 eval_metrics_top_n, 
                 clicked_jobs_state, 
                 eval_sessions_metrics_log,
                 sessions_negative_jobs_log,
                 sessions_model_recommendations_log,
                 pretrained_job_content_embeddings,
                 job_metadata,
                 user_metadata,
                 eval_baseline_classifiers=[],
                 attention_log=[],
                 trained_session_embeddings=[]
                 ):
        self.mode = mode
        self.model = model        
        self.eval_metrics_top_n = eval_metrics_top_n
                
        self.clicked_jobs_state = clicked_jobs_state
        self.eval_sessions_metrics_log = eval_sessions_metrics_log
        self.sessions_negative_jobs_log = sessions_negative_jobs_log
        self.sessions_model_recommendations_log = sessions_model_recommendations_log

        self.pretrained_job_content_embeddings = pretrained_job_content_embeddings
        self.job_metadata = job_metadata
        self.user_metadata = user_metadata


        self.baselines_classifiers = list([clf['recommender'](
            self.clicked_jobs_state,
            clf['eval_baseline_params'],
            JobsStateUpdaterHook.create_eval_metrics(
                self.eval_metrics_top_n,
                self.pretrained_job_content_embeddings,
                self.job_metadata,
                self.user_metadata,
                self.clicked_jobs_state)) for clf in eval_baseline_classifiers])

        # Store session_id
        self.session_id_log = []
        # Store user_id
        self.user_id_log = []
        # Store applied jobs
        self.clicked_job_log = []
        # Store next job
        self.next_job_label_log = []
        # Store last job 
        self.last_job_label_log = []
        # Store attention scores
        self.attention_log = []
        # Store session representations
        self.trained_session_embeddings = []

    def begin(self):        
        if self.mode == tf.estimator.ModeKeys.EVAL:

            tf.logging.info("Saving jobs state checkpoint from train")
            #Save state of jobs popularity and recency from train loop, to restore after evaluation finishes
            self.clicked_jobs_state.save_state_checkpoint()  
            
            #Resets streaming metrics
            self.eval_streaming_metrics_last = {}            
            for clf in self.baselines_classifiers:
                clf.reset_eval_metrics()

    
            self.streaming_metrics = JobsStateUpdaterHook.create_eval_metrics(
                self.eval_metrics_top_n, 
                self.pretrained_job_content_embeddings,
                self.job_metadata,
                self.user_metadata,
                self.clicked_jobs_state)
            #self.metrics_by_session_pos = StreamingMetrics(topn=self.metrics_top_n)
                
            self.stats_logs = []


    #Runs before every batch
    def before_run(self, run_context): 
        fetches = {
        'clicked_jobs': self.model.job_clicked,
        'application_dates': self.model.application_dates,
        'session_start': self.model.session_start,
        'next_job_labels': self.model.next_job_label,
        'last_job_label': self.model.label_last_job,                   
        'session_id': self.model.session_id,
        'session_start': self.model.session_start,
        'user_ids': self.model.user_ids
        }

        if self.mode == tf.estimator.ModeKeys.EVAL:
            fetches['predicted_job_ids'] = self.model.predicted_job_ids
            fetches['eval_batch_negative_jobs'] = self.model.batch_negative_jobs


        if self.mode == tf.estimator.ModeKeys.EVAL:            
            fetches['batch_jobs_count'] = self.model.batch_jobs_count
            fetches['batch_unique_jobs_count'] = self.model.batch_unique_jobs_count
            fetches['batch_jobcity_count'] = self.model.batch_jobcity_count
            fetches['batch_unique_jobcity_count'] = self.model.batch_unique_jobcity_count
            fetches['batch_jobstate_count'] = self.model.batch_jobstate_count
            fetches['batch_unique_jobstate_count'] = self.model.batch_unique_jobstate_count
            fetches['batch_usercity_count'] = self.model.batch_usercity_count
            fetches['batch_unique_usercity_count'] = self.model.batch_unique_usercity_count
            fetches['batch_userstate_count'] = self.model.batch_userstate_count
            fetches['batch_unique_userstate_count'] = self.model.batch_unique_userstate_count
            fetches['batch_same_city_count'] = self.model.batch_same_city_count
            fetches['batch_same_state_count'] = self.model.batch_same_state_count

            fetches['hitrate_at_n'] = self.model.recall_at_n_update_op
            fetches['mrr_at_n'] = self.model.mrr_update_op
            fetches['ndcg_at_n'] = self.model.ndcg_at_n_mean_update_op  
            fetches['predicted_job_probs'] = self.model.predicted_job_probs

            fetches['attention'] = self.model.attention
            fetches['session_representations'] = self.model.session_representations





        feed_dict = {
            self.model.jobs_recent_pop_norm: self.clicked_jobs_state.get_jobs_recent_pop_norm(),            
            self.model.pop_recent_jobs_buffer:  self.clicked_jobs_state.get_recent_clicks_buffer(), 
            #Passed as placeholder (and not as a constant) to avoid been saved in checkpoints
            self.model.pretrained_job_content_embeddings: self.pretrained_job_content_embeddings           
        }         

        #Passed as placeholder (and not as a constant) to avoid been saved in checkpoints
    
        for feature_name in self.job_metadata:
            feed_dict[self.model.job_metadata[feature_name]] = self.job_metadata[feature_name]

        for feature_name in self.user_metadata:
            feed_dict[self.model.user_metadata[feature_name]] = self.user_metadata[feature_name]


        return tf.estimator.SessionRunArgs(fetches=fetches,feed_dict=feed_dict)


    #Runs after every batch
    def after_run(self, run_context, run_values):
        clicked_jobs = run_values.results['clicked_jobs']
        application_dates = np.squeeze(run_values.results['application_dates'], axis=-1)
        next_job_labels = run_values.results['next_job_labels']
        last_job_label = run_values.results['last_job_label'] 

        users_ids = run_values.results['user_ids']
        sessions_ids = run_values.results['session_id']
        
        if self.mode == tf.estimator.ModeKeys.EVAL:
            predicted_job_ids = run_values.results['predicted_job_ids']
            #tf.logging.info('predicted_item_ids (shape): {}'.format(predicted_item_ids.shape))  
            eval_batch_negative_jobs = run_values.results['eval_batch_negative_jobs'] 


        if self.mode == tf.estimator.ModeKeys.EVAL:
            self.eval_streaming_metrics_last = {}
            #self.eval_streaming_metrics_last['hitrate_at_1'] = run_values.results['hitrate_at_1']
            self.eval_streaming_metrics_last['hitrate_at_n'] = run_values.results['hitrate_at_n']
            self.eval_streaming_metrics_last['mrr_at_n'] = run_values.results['mrr_at_n']
            self.eval_streaming_metrics_last['ndcg_at_n'] = run_values.results['ndcg_at_n']

            predicted_job_probs = run_values.results['predicted_job_probs']

            if self.sessions_negative_jobs_log != None:
                #Acumulating session negative items, to allow evaluation comparison with benchmarks outsite the framework (e.g. Matrix Factorization) 
                for session_id, labels, neg_jobs in zip(sessions_ids, next_job_labels, eval_batch_negative_jobs):
                    self.sessions_negative_jobs_log.append(
                    {'session_id': str(session_id), #Convert numeric session_id to str because large ints are not serializable
                    'negative_jobs': list([neg_jobs_click for label, neg_jobs_click in zip(labels.tolist(), neg_jobs.tolist()) if label != 0])})


            if self.sessions_model_recommendations_log != None:
                predicted_job_probs = run_values.results['predicted_job_probs']
                predicted_job_probs_rounded = predicted_job_probs.round(decimals=7)

                jobs_recent_pop_norm = self.clicked_jobs_state.get_jobs_recent_pop_norm()

                #Acumulating CHAMELEON predictions, labels, scores, accuracy to allow greed re-ranking approachs (e.g. MMR)
                for session_id, labels, pred_job_ids, pred_job_probs in zip(sessions_ids, next_job_labels, predicted_job_ids, predicted_job_probs_rounded): 
                    #Reducing the precision to 5 decimals for serialization
                    pred_job_norm_pops = jobs_recent_pop_norm[pred_job_ids].round(decimals=7)

                    labels_filtered = []
                    pred_job_ids_filtered = []
                    pred_job_probs_filtered = []
                    pred_job_norm_pops_filtered = []

                    for label, pred_job_ids_click, pred_job_probs_click, pred_job_norm_pops_click in zip(labels.tolist(), pred_job_ids.tolist(), pred_job_probs.tolist(), pred_job_norm_pops.tolist()):
                        if label != 0:
                            labels_filtered.append(str(label))
                            pred_job_ids_filtered.append(pred_job_ids_click)
                            pred_job_probs_filtered.append(pred_job_probs_click)
                            pred_job_norm_pops_filtered.append(pred_job_norm_pops_click)


                    to_append = {
                    'session_id': str(session_id), #Convert numeric session_id to str because large ints are not serializable
                    'next_click_labels': labels_filtered,
                    'predicted_job_ids': pred_job_ids_filtered,
                    'predicted_job_probs': pred_job_probs_filtered,
                    'predicted_job_norm_pop': pred_job_norm_pops_filtered}


                    self.sessions_model_recommendations_log.append(to_append) 



            batch_stats = {#'eval_sampled_negative_items': eval_batch_negative_items.shape[1],
                           'batch_jobs_count': run_values.results['batch_jobs_count'],
                           'batch_unique_jobs_count': run_values.results['batch_unique_jobs_count'],
                           'batch_jobcity_count': run_values.results['batch_jobcity_count'],
                           'batch_unique_jobcity_count': run_values.results['batch_unique_jobcity_count'],
                           'batch_jobstate_count': run_values.results['batch_jobstate_count'],
                           'batch_unique_jobstate_count': run_values.results['batch_unique_jobstate_count'],
                           'batch_usercity_count': run_values.results['batch_usercity_count'],
                           'batch_unique_usercity_count': run_values.results['batch_unique_usercity_count'],
                           'batch_userstate_count': run_values.results['batch_userstate_count'],
                           'batch_unique_userstate_count': run_values.results['batch_unique_userstate_count'],
                           'batch_sessions_count': len(sessions_ids),
                           'batch_same_city_count': run_values.results['batch_same_city_count'],
                           'batch_same_state_count': run_values.results['batch_same_state_count'],
                           }

            self.stats_logs.append(batch_stats)
            tf.logging.info('batch_stats: {}'.format(batch_stats))

            preds_norm_pop = self.clicked_jobs_state.get_jobs_recent_pop_norm()[predicted_job_ids]
            labels_norm_pop = self.clicked_jobs_state.get_jobs_recent_pop_norm()[next_job_labels]


            #Computing metrics for this neural model
            update_metrics(predicted_job_ids, next_job_labels, clicked_jobs, self.streaming_metrics, recommender='PAN') 
            model_metrics_values = compute_metrics_results(self.streaming_metrics, recommender='PAN') 

            self.eval_streaming_metrics_last = merge_two_dicts(self.eval_streaming_metrics_last, model_metrics_values)


            start_eval = time()

            #Computing metrics for baseline recommenders
            for clf in self.baselines_classifiers:
                self.evaluate_and_update_streaming_metrics_last(clf, users_ids, clicked_jobs, next_job_labels, eval_batch_negative_jobs)
            tf.logging.info('Total elapsed time evaluating benchmarks: {}'.format(time() - start_eval))            

            tf.logging.info('Finished benchmarks evaluation')

            self.session_id_log.append(run_values.results['session_id'])
            self.user_id_log.append(run_values.results['user_ids'])
            self.clicked_job_log.append(run_values.results['clicked_jobs'])
            self.next_job_label_log.append(run_values.results['next_job_labels'])
            self.last_job_label_log.append(run_values.results['last_job_label'])
            self.attention_log.append(run_values.results['attention'])
            #tf.logging.info('attention_log: {}'.format(self.attention_log))
            self.trained_session_embeddings.append(run_values.results['session_representations'])


        #Training baseline recommenders
        for clf in self.baselines_classifiers:
            #It is required that session_ids are sorted by time (ex: first_timestamp+hash_session_id), so that recommenders that trust in session_id to sort by recency work (e.g. V-SkNN)
            clf.train(users_ids, sessions_ids, clicked_jobs, next_job_labels)

        # Concatenating all clicked jobs in the batch (including last label)
        batch_clicked_jobs = np.concatenate([clicked_jobs, last_job_label], axis=1)
        #Flattening values and removing padding items (zeroes) 
        batch_clicked_jobs_flatten = batch_clicked_jobs.reshape(-1)
        batch_clicked_jobs_nonzero = batch_clicked_jobs_flatten[np.nonzero(batch_clicked_jobs_flatten)]

        #As timestamp of last clicks are not available for each session, assuming they are the same than previous session click
        last_application_date_batch = np.max(application_dates, axis=1).reshape(-1,1)
        batch_application_date = np.concatenate([application_dates,last_application_date_batch], axis=1)
        #Flattening values and removing padding items (zeroes)        
        batch_application_date_flatten = batch_application_date.reshape(-1)        
        batch_application_date_nonzero = batch_application_date_flatten[np.nonzero(batch_application_date_flatten)]

        #Updating jobs state
        self.clicked_jobs_state.update_jobs_state(batch_clicked_jobs_nonzero, batch_application_date_nonzero)        
        self.clicked_jobs_state.update_jobs_coocurrences(batch_clicked_jobs)

       

    def end(self, session=None):
        if self.mode == tf.estimator.ModeKeys.EVAL:    
            #avg_neg_items = np.mean([x['eval_sampled_negative_items'] for x in self.stats_logs])
            #self.eval_streaming_metrics_last['avg_eval_sampled_neg_items'] = avg_neg_items
            clicks_count = np.sum([x['batch_jobs_count'] for x in self.stats_logs])
            self.eval_streaming_metrics_last['clicks_count'] = clicks_count

            sessions_count = np.sum([x['batch_sessions_count'] for x in self.stats_logs])
            self.eval_streaming_metrics_last['sessions_count'] = sessions_count


            self.eval_sessions_metrics_log.append(self.eval_streaming_metrics_last)
            eval_metrics_str = '\n'.join(["'{}':\t{}".format(metric, value) for metric, value in sorted(self.eval_streaming_metrics_last.items())])
            tf.logging.info("Evaluation metrics: [{}]".format(eval_metrics_str))

            #Logs stats for time delay for the first job recommendation since its first click
            #self.clicked_items_state.log_stats_time_for_first_rec()

            tf.logging.info("Restoring jobs state checkpoint from train")
            #Restoring the original state of items popularity and recency state from train loop
            self.clicked_jobs_state.restore_state_checkpoint()


            outputs = {'session_ids': self.session_id_log, 'user_ids': self.user_id_log, 'clicked_jobs': self.clicked_job_log, 'next_job_labels': self.next_job_label_log, 'last_job_label': self.last_job_label_log, 'attention': self.attention_log, 'session_representations': self.trained_session_embeddings}
            serialize('outputs', outputs)

           
       

    def evaluate_and_update_streaming_metrics_last(self, clf,  users_ids, clicked_jobs, next_job_labels, eval_batch_negative_jobs):
        tf.logging.info('Evaluating benchmark: {}'.format(clf.get_description())) 
        clf_metrics = clf.evaluate(users_ids, clicked_jobs, next_job_labels, topk=self.eval_metrics_top_n, eval_negative_jobs=eval_batch_negative_jobs)
        self.eval_streaming_metrics_last = merge_two_dicts(self.eval_streaming_metrics_last, clf_metrics)



    @staticmethod
    def create_eval_metrics(top_n, pretrained_job_content_embeddings, job_metadata, user_metadata, clicked_jobs_state):
        recent_clicks_buffer = clicked_jobs_state.get_recent_clicks_buffer() 
        eval_metrics = [metric(topn=top_n) for metric in [HitRate, MRR, NDCG]]
        return eval_metrics

  





           
          


  
    
       





       



