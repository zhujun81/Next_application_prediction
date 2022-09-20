import tensorflow as tf
print(tf.__version__)
tf.compat.v1.enable_eager_execution()
print(tf.executing_eagerly())
from tensorflow.python.ops import array_ops
from tensorflow.contrib.layers import xavier_initializer, variance_scaling_initializer

from utils import get_tf_dtype, get_embedding_size
from evaluation import tf_ndcg_at_k


JOB_REQ_FEATURES = ['JobID', 'StartDate', 'EndDate']
SESSION_REQ_SEQ_FEATURES = ['Job_clicked', 'ApplicationDate', 'JobCity', 'JobState', 'JobCountry', 'UserCity', 'UserState', 'UserCountry', #'UserDegree', 'UserMajor'
]



class PAN():
    def __init__(
        self, mode, 
        inputs, 
        labels,  
        session_features_config,
        job_features_config, 
        user_features_config,
        batch_size, 
        lr=0.01, # Lerning Rate
        reg_l2_rate=0.0,  # L2 regularization
        dropout_keep_prob=0.0, 
        softmax_temperature=1.0, # Initial value for temperature for softmax
        pretrained_job_content_embeddings = None,
        pretrained_job_embedding_size=300,
        job_metadata = None,
        user_metadata = None,
        negative_samples=10,  # Total negative samples for training/eval
        negative_sample_from_buffer=20, # Training(10)/Eval(50) Negative samples from recent clicks buffer
        rnn_num_layers=1,# Number of of RNN layers 
        rnn_units=256, # Number of units of RNN cell          
        recent_clicks_buffer_max_size = 1000, # Maximum size of recent clicks buffer
        recent_clicks_for_normalization = 1000, # Number of recent clicks to normalize recency and populary  novelty) dynamic features
        metrics_top_n=5,
        max_cardinality_for_ohe=10,
        internal_features_config={
        'job_content_embeddings': True,
        'job_clicked_embeddings': True},
        plot_histograms=False
        ):


        ### Training parameters
        self.lr = lr 
        self.reg_l2_rate = reg_l2_rate
        self.dropout_keep_prob = dropout_keep_prob
        self.softmax_temperature = tf.constant(softmax_temperature, dtype=tf.float32, name='softmax_temperature')
        self.is_training = (mode == tf.estimator.ModeKeys.TRAIN) 

        ### Pretrained job embedding
        self.pretrained_job_embedding_size = pretrained_job_embedding_size
        self.max_cardinality_for_ohe = max_cardinality_for_ohe

        ### Negative samples
        self.negative_samples = negative_samples 
        self.negative_sample_from_buffer = negative_sample_from_buffer
        
        ### Recent applications buffer
        self.recent_clicks_buffer_max_size = recent_clicks_buffer_max_size
        self.recent_clicks_for_normalization = recent_clicks_for_normalization

        ### RNN parameters
        self.rnn_num_layers = rnn_num_layers
        self.rnn_units = rnn_units

        ### Evaluation
        self.metrics_top_n = metrics_top_n

        self.plot_histograms = plot_histograms

        self.internal_features_config = internal_features_config
        self.session_features_config = session_features_config
        self.job_features_config = job_features_config
        self.user_features_config = user_features_config
        
        self.recent_clicks_for_normalization = recent_clicks_for_normalization
     
       
        

        with tf.compat.v1.variable_scope("job_content_embeddings"):
            self.job_metadata = {}
            with tf.device('/cpu:0'):
                #Converting job metadata feature vectors to constants in the graph, to avoid many copies (is saved with the graph)
                for feature_name in job_metadata:
                    self.job_metadata[feature_name] = tf.compat.v1.placeholder(name="job_metadata",
                                                                     shape=job_metadata[feature_name].shape, 
                                                                     dtype=get_tf_dtype(job_features_config[feature_name]['dtype']))
            
            self.job_vocab_size = job_features_config['JobID']['cardinality']
            
            #To run on local machine (GPU card with 4 GB RAM), keep Content Article Embeddings constant in CPU memory
            with tf.device('/cpu:0'):
                #Expects vectors within the range [-0.1, 0.1] (min-max scaled) for compatibility with other input features
                self.pretrained_job_content_embeddings = tf.compat.v1.placeholder(name="pretrained_job_content_embeddings", shape=pretrained_job_content_embeddings.shape, dtype=tf.float32)

        
        with tf.compat.v1.variable_scope("job_status"):
            self.pop_recent_jobs_buffer_by_state = {}
            with tf.device('/cpu:0'):
                self.jobs_recent_pop_norm = tf.compat.v1.placeholder(name="jobs_recent_pop_norm", shape=[self.job_vocab_size], dtype=tf.float32)


            self.pop_recent_jobs_buffer = tf.compat.v1.placeholder(name="pop_recent_jobs_buffer", shape=[self.recent_clicks_buffer_max_size], dtype=tf.int64)          
            
            #for state_name in job_metadata['JobState']:
                #self.pop_recent_jobs_buffer_by_state[state_name] = tf.placeholder(name="pop_recent_jobs_buffer", shape=[self.recent_clicks_buffer_max_size], dtype=tf.int64)   

            #for state_name in job_metadata['JobState']:
                #tf.summary.scalar('unique_jobs_clicked_recently', family='stats', tensor=tf.shape(tf.unique(self.pop_recent_jobs_buffer_by_state[state_name])[0])[0])   
             
            tf.compat.v1.summary.scalar('unique_jobs_clicked_recently_for_normalization', family='stats', tensor=tf.shape(tf.unique(self.pop_recent_jobs_buffer[:self.recent_clicks_for_normalization])[0])[0])


        with tf.compat.v1.variable_scope("user_metadata_features"):
            self.user_metadata = {}
            with tf.device('/cpu:0'):
                #Converting job metadata feature vectors to constants in the graph, to avoid many copies (is saved with the graph)
                for feature_name in user_metadata:
                    self.user_metadata[feature_name] = tf.compat.v1.placeholder(name="user_metadata",
                                                                     shape=user_metadata[feature_name].shape, 
                                                                     dtype=get_tf_dtype(user_features_config[feature_name]['dtype']))
            
            self.user_vocab_size = user_features_config['UserID']['cardinality']
            

        #PS: variance_scaling_initializer() is recommended for RELU activations in https://arxiv.org/abs/1502.01852
        #whilst xavier_initializer is recommended for tanh activations
        with tf.compat.v1.variable_scope("main", initializer=xavier_initializer()):
            with tf.compat.v1.variable_scope("inputs"):
                job_clicked = inputs['Job_clicked']
                self.job_clicked = job_clicked

                #Control features (ensuring that they keep two dims even when the batch has only one session)
                user_ids = inputs['UserID']
                user_cities = inputs['UserCity']
                user_states = inputs['UserState']

                self.user_ids = user_ids
                self.user_cities = user_cities
                self.user_states = user_states
                self.session_id = inputs['SessionID']
                self.session_start = inputs["SessionStart"]

                
                seq_lengths = inputs['SessionSize'] - 1 #Ignoring last click only as label
                self.seq_lengths = seq_lengths
                
                
                #Creates the sessions mask and ensure that rank will be 2 (even when this batch size is 1)
                self.job_clicked_mask = tf.sequence_mask(seq_lengths)
                

                application_dates = tf.expand_dims(inputs['ApplicationDate'], -1)
                self.application_dates = application_dates
                max_application_date = tf.reduce_max(application_dates)

                
                #Retrieving last label of the sequence
                label_last_job = labels['label_last_job'] 
                self.label_last_job = label_last_job
                all_clicked_jobs = tf.concat([job_clicked, label_last_job], axis=1)
                

                #Labels            
                next_job_label = labels['label_next_job']
                self.next_job_label = next_job_label
                
                batch_max_session_length = tf.shape(next_job_label)[1] 
                batch_current_size = array_ops.shape(next_job_label)[0]
            

            
            with tf.compat.v1.variable_scope("batch_stats"):
                #batch_items = self.get_masked_seq_values(inputs['item_clicked']) 
                #Known bug: The article_id 0 will not be considered as negative sample, because padding values also have value 0 
                batch_jobs_nonzero = tf.boolean_mask(all_clicked_jobs, tf.cast(tf.sign(all_clicked_jobs), tf.bool))
                batch_jobs_count = tf.shape(batch_jobs_nonzero)[0]
                self.batch_jobs_count = batch_jobs_count
                

                ### Jobs
                batch_unique_jobs, _ = tf.unique(batch_jobs_nonzero)
                batch_unique_jobs_count = tf.shape(batch_unique_jobs)[0]
                self.batch_unique_jobs_count = batch_unique_jobs_count
         
                self.batch_jobcity = tf.gather(self.job_metadata['JobCity'], batch_jobs_nonzero)
                batch_jobcity_count = tf.shape(self.batch_jobcity)[0]
                self.batch_jobcity_count = batch_jobcity_count

                batch_unique_jobcity, _ = tf.unique(self.batch_jobcity)
                batch_unique_jobcity_count = tf.shape(batch_unique_jobcity)[0]
                self.batch_unique_jobcity_count = batch_unique_jobcity_count

                self.batch_jobstate = tf.gather(self.job_metadata['JobState'], batch_jobs_nonzero)
                batch_jobstate_count = tf.shape(self.batch_jobstate)[0]
                self.batch_jobstate_count = batch_jobstate_count

                batch_unique_jobstate, _ = tf.unique(self.batch_jobstate)
                batch_unique_jobstate_count = tf.shape(batch_unique_jobstate)[0]
                self.batch_unique_jobstate_count = batch_unique_jobstate_count


                ### User

                batch_usercity = tf.gather(self.user_metadata['UserCity'], user_ids)
                batch_usercity_count = tf.shape(batch_usercity)[0]
                self.batch_usercity_count = batch_usercity_count

                batch_unique_usercity, _ = tf.unique(tf.squeeze(batch_usercity, axis = -1))
                batch_unique_usercity_count = tf.shape(batch_unique_usercity)[0]
                self.batch_unique_usercity_count = batch_unique_usercity_count

                batch_userstate = tf.gather(self.user_metadata['UserState'], user_ids)
                batch_userstate_count = tf.shape(batch_userstate)[0]
                self.batch_userstate_count = batch_userstate_count

                batch_unique_userstate, _ = tf.unique(tf.squeeze(batch_userstate, axis = -1))
                batch_unique_userstate_count = tf.shape(batch_unique_userstate)[0]
                self.batch_unique_userstate_count = batch_unique_userstate_count


                ### Job and User
                batch_same_city = tf.sets.intersection(batch_unique_jobcity[None,:], batch_unique_usercity[None,:], validate_indices=True)
                batch_same_city_count = tf.shape(batch_same_city)[1]
                self.batch_same_city_count = batch_same_city_count

                batch_same_state = tf.sets.intersection(batch_unique_jobstate[None,:], batch_unique_userstate[None,:], validate_indices=True)
                batch_same_state_count = tf.shape(batch_same_state)[1]
                self.batch_same_state_count = batch_same_state_count
              
                
                tf.compat.v1.summary.scalar('batch_jobs', family='stats', tensor=batch_jobs_count)
                tf.compat.v1.summary.scalar('batch_unique_jobs', family='stats', tensor=batch_unique_jobs_count)
                tf.compat.v1.summary.scalar('batch_jobcity', family='stats', tensor=batch_jobcity_count)
                tf.compat.v1.summary.scalar('batch_unique_jobcity', family='stats', tensor=batch_unique_jobcity_count)
                tf.compat.v1.summary.scalar('batch_jobstate', family='stats', tensor=batch_jobstate_count)
                tf.compat.v1.summary.scalar('batch_unique_jobstate', family='stats', tensor=batch_unique_jobstate_count)
                tf.compat.v1.summary.scalar('batch_usercity', family='stats', tensor=batch_usercity_count)
                tf.compat.v1.summary.scalar('batch_unique_usercity', family='stats', tensor=batch_unique_usercity_count)
                tf.compat.v1.summary.scalar('batch_userstate', family='stats', tensor=batch_userstate_count)
                tf.compat.v1.summary.scalar('batch_unique_userstate', family='stats', tensor=batch_unique_userstate_count)
               
               
            
            with tf.compat.v1.variable_scope("neg_samples"):

                #Samples from recent jobs buffer
                # 对于每个session 都是一样的 additional_samples
                negative_sample_recently_clicked_ids = self.get_sample_from_recently_clicked_jobs_buffer(self.negative_sample_from_buffer)            
                
                #Samples from other sessions in the same mini-batch, negative_sample_recently_clicked_ids as additional samples
                # 把user_states考虑进去
                # batch_negative_jobs shape=(?, ?, ?)
                batch_negative_jobs = self.get_batch_negative_samples(all_clicked_jobs, batch_usercity, batch_userstate, additional_samples=negative_sample_recently_clicked_ids, num_negative_samples=self.negative_samples)

                #Ignoring last elements from second dimension, as they refer to the last labels concatenated with all_clicked_jobs just to ignore them in negative samples
                # batch_negative_jobs shape=(?, ?, ?)
                batch_negative_jobs = batch_negative_jobs[:,:-1,:] 
                self.batch_negative_jobs = batch_negative_jobs


            with tf.compat.v1.variable_scope("jobs_contextual_features"):
                ##### Input jobs
                input_jobs_features = self.get_job_features(job_clicked, 'clicked')                                           
                if self.plot_histograms:
                    tf.summary.histogram("input_jobs_features", input_jobs_features)

                input_jobs_features = self.scale_center_features(input_jobs_features)
                if self.plot_histograms:
                    tf.summary.histogram("input_jobs_features", input_jobs_features)
                
                # (?, ?, 576)
                input_jobs_features = tf.layers.dropout(input_jobs_features, rate=1.0-self.dropout_keep_prob, training=self.is_training)

                ##### Positive jobs
                positive_jobs_features = self.get_job_features(next_job_label, 'positive')
                if self.plot_histograms:
                    tf.summary.histogram("positive_jobs_features", positive_jobs_features)
                positive_jobs_features = self.scale_center_features(positive_jobs_features)
                if self.plot_histograms:
                    tf.summary.histogram("positive_jobs_features", positive_jobs_features)
                # (?, ?, 576)
                positive_jobs_features = tf.layers.dropout(positive_jobs_features,rate=1.0-self.dropout_keep_prob, training=self.is_training)
                
                ##### Negative jobs
                negative_jobs_features = self.get_job_features(batch_negative_jobs, 'negative')
                if self.plot_histograms:
                    tf.summary.histogram("negative_jobs_features", negative_jobs_features)
                negative_jobs_features = self.scale_center_features(negative_jobs_features, begin_norm_axis=3)
                if self.plot_histograms:
                    tf.summary.histogram("negative_jobs_features", negative_jobs_features)
                # (?, ?, 576)
                negative_jobs_features = tf.layers.dropout(negative_jobs_features, rate=1.0-self.dropout_keep_prob, training=self.is_training)
            

            # layer 1: Leaky ReLU
            with tf.compat.v1.variable_scope("PreJobRep"):
                PreJobRep_dense = tf.compat.v1.layers.Dense(
                    self.pretrained_job_embedding_size,
                    activation=tf.nn.leaky_relu, 
                    kernel_initializer=variance_scaling_initializer(),
                    kernel_regularizer=tf.contrib.layers.l2_regularizer(self.reg_l2_rate),
                    name="Prejob_representation") 

                input_contextual_job_embedding_pre = PreJobRep_dense(input_jobs_features)

                # layer 2: tanh
                JobRep_dense = tf.compat.v1.layers.Dense(self.pretrained_job_embedding_size,
                                            activation=tf.nn.tanh, 
                                            kernel_regularizer=tf.contrib.layers.l2_regularizer(self.reg_l2_rate),
                                            name="personalized_job_representation"
                                       )


            with tf.compat.v1.variable_scope("contextual_job_embedding"):
                with tf.compat.v1.variable_scope("input"):
                    contextual_job_embedding = JobRep_dense(input_contextual_job_embedding_pre)
                    
                    if self.plot_histograms:
                        tf.summary.histogram("input_contextual_job_embedding", contextual_job_embedding)
                
                
                with tf.compat.v1.variable_scope("positive"): 
                    positive_contextual_job_embedding = JobRep_dense(PreJobRep_dense(positive_jobs_features))
                    if self.plot_histograms:
                        tf.summary.histogram("positive_contextual_job_embedding", positive_contextual_job_embedding)
                
                
                with tf.compat.v1.variable_scope("negative"): 
                    negative_contextual_job_embedding = JobRep_dense(PreJobRep_dense(negative_jobs_features))
                    if self.plot_histograms:
                        tf.summary.histogram("negative_contextual_job_embedding", negative_contextual_job_embedding)
            
       
          



            #Building RNN
            # rnn_outputs (?, ?, 256)
            #rnn_outputs = self.build_rnn(contextual_job_embedding, seq_lengths, rnn_units=self.rnn_units)
            #print('LSTMCell')
            att, rnn_outputs = self.build_pan(contextual_job_embedding, user_ids, seq_lengths)
            print('PAN')
            # The RNN layer is followed by a sequence of two feed-forward layers, with Leaky ReLU and tanh activation functions
            with tf.compat.v1.variable_scope("session_representation"): 
                #WARNING: Must keep these variables under the same variable scope, to avoid leaking the positive item to the network (probably due to normalization)
                with tf.compat.v1.variable_scope("user_contextual_features"):
                    # (?, ?, 94)
                    user_context_features = self.get_features(inputs, features_config=self.session_features_config['sequence_features'], features_to_ignore=SESSION_REQ_SEQ_FEATURES)
                    #print('user_context_features', user_context_features.shape)
                    #If there is no user contextual features, creates a dummy variable to not break following concats
                    if user_context_features != None:
                        if self.plot_histograms:
                            tf.summary.histogram("user_context_features", user_context_features)
                    else:
                        #Dummy tensor with zeroed values
                        user_context_features = tf.zeros_like(tf.expand_dims(job_clicked, -1), dtype=tf.float32)
                

                output_user_rnn_concat = tf.concat([user_context_features, rnn_outputs], axis=2)

                rnn_outputs_fc1 = tf.layers.dense(
                    output_user_rnn_concat,
                    512,
                    activation=tf.nn.leaky_relu, 
                    kernel_initializer=variance_scaling_initializer(),
                    kernel_regularizer=tf.contrib.layers.l2_regularizer(self.reg_l2_rate),
                    name="FC1")
                rnn_outputs_fc1_dropout = tf.layers.dropout(inputs=rnn_outputs_fc1, rate=1.0-self.dropout_keep_prob, training=self.is_training)
                
               
                rnn_outputs_fc2 = tf.layers.dense(
                    rnn_outputs_fc1_dropout, 
                    pretrained_job_embedding_size,
                    activation=tf.nn.tanh, 
                    name='FC2', 
                    kernel_regularizer=tf.contrib.layers.l2_regularizer(self.reg_l2_rate))
                
                if self.plot_histograms:
                    tf.summary.histogram("rnn_outputs_fc2", rnn_outputs_fc2)


                # Store 
                self.attention = att
                self.session_representations = rnn_outputs_fc2

            
            with tf.compat.v1.variable_scope("predicted_contextual_job_embedding"): 
                #Continuing with DSSM losss
                #Apply l2-norm to be able to compute cosine similarity by matrix multiplication
                #predicted_contextual_item_embedding = tf.nn.l2_normalize(rnn_outputs_fc2, axis=-1)
                predicted_contextual_job_embedding = rnn_outputs_fc2
                if self.plot_histograms:
                    tf.summary.histogram("predicted_contextual_job_embedding", predicted_contextual_job_embedding)


            # Output: relevance score
            # 1. element-wise product
            # 2. A sequence of 4 feed-forward layers with Leaky ReLU, 12
            # 3. softmax
            with tf.compat.v1.variable_scope("recommendations_ranking"): 
                matching_dense_layer_1 = tf.compat.v1.layers.Dense(
                    128,
                    activation=tf.nn.leaky_relu, 
                    kernel_initializer=variance_scaling_initializer(),
                    kernel_regularizer=tf.contrib.layers.l2_regularizer(self.reg_l2_rate),
                    name="matching_dense_layer_1")
               
                matching_dense_layer_2 = tf.compat.v1.layers.Dense(
                    64,
                    activation=tf.nn.leaky_relu, 
                    kernel_initializer=variance_scaling_initializer(),
                    kernel_regularizer=tf.contrib.layers.l2_regularizer(self.reg_l2_rate),
                    name="matching_dense_layer_2")

                matching_dense_layer_3 = tf.compat.v1.layers.Dense(
                    32,
                    activation=tf.nn.leaky_relu, 
                    kernel_initializer=variance_scaling_initializer(), 
                    kernel_regularizer=tf.contrib.layers.l2_regularizer(self.reg_l2_rate),
                    name="matching_dense_layer_3")

                matching_dense_layer_4 = tf.compat.v1.layers.Dense(
                    1,
                    activation=None, 
                    kernel_initializer=tf.compat.v1.initializers.lecun_uniform(),
                    kernel_regularizer=tf.contrib.layers.l2_regularizer(self.reg_l2_rate),
                    name="matching_dense_layer_4")

                
                with tf.compat.v1.variable_scope("cos_sim_positive"):
                    positive_multiplied_embeddings = tf.multiply(positive_contextual_job_embedding, predicted_contextual_job_embedding)
                    if self.plot_histograms:
                        tf.summary.histogram("train/positive_multiplied_embeddings", positive_multiplied_embeddings)
                    cos_sim_positive = matching_dense_layer_4(matching_dense_layer_3(matching_dense_layer_2(matching_dense_layer_1(positive_multiplied_embeddings))))
                    if self.plot_histograms:
                        tf.summary.histogram("train/cos_sim_positive", values=tf.boolean_mask(cos_sim_positive, tf.cast(tf.sign(next_job_label), tf.bool)))
                
                with tf.compat.v1.variable_scope("cos_sim_negative"):
                    negative_multiplied_embeddings = tf.multiply(negative_contextual_job_embedding, tf.expand_dims(predicted_contextual_job_embedding, 2))
                    if self.plot_histograms:
                        tf.summary.histogram("train/negative_multiplied_embeddings", negative_multiplied_embeddings)
                    cos_sim_negative = matching_dense_layer_4(matching_dense_layer_3(matching_dense_layer_2(matching_dense_layer_1(negative_multiplied_embeddings))))
                    cos_sim_negative = tf.squeeze(cos_sim_negative,  axis=-1)
                    if self.plot_histograms:
                        tf.summary.histogram("train/cos_sim_negative", values=tf.boolean_mask(cos_sim_negative, tf.cast(tf.sign(next_job_label), tf.bool)))
                
                with tf.compat.v1.variable_scope("softmax_function"):
                    #Concatenating cosine similarities (positive + K sampled negative)
                    cos_sim_concat = tf.concat([cos_sim_positive, cos_sim_negative], axis=2)                    
                    
                    #Computing softmax over cosine similarities
                    cos_sim_concat_scaled = cos_sim_concat / self.softmax_temperature
                    jobs_prob = tf.nn.softmax(cos_sim_concat_scaled) 
                    neg_jobs_prob = tf.nn.softmax(cos_sim_negative / self.softmax_temperature)
                
                if mode == tf.estimator.ModeKeys.EVAL:
                    pos_neg_jobs_ids = tf.concat([tf.expand_dims(next_job_label, -1), batch_negative_jobs], 2)
                    predicted_job_ids, predicted_job_probs = self.rank_jobs_by_predicted_prob(pos_neg_jobs_ids, jobs_prob)
                    self.predicted_job_ids = predicted_job_ids
                    self.predicted_job_probs = predicted_job_probs
    
                
                if (mode == tf.estimator.ModeKeys.EVAL):
                    #Computing evaluation metrics
                    self.define_eval_metrics(next_job_label, predicted_job_ids)


            
            with tf.compat.v1.variable_scope("loss"):
                #Computing batch loss
                loss_mask = tf.to_float(self.job_clicked_mask)

                #Computing the probability of the positive item (label)
                positive_prob = jobs_prob[:,:,0]
                negative_probs = jobs_prob[:,:,1:]
                
                #Summary of first element of the batch sequence (because others might be masked)
                if self.plot_histograms:
                    tf.summary.histogram("positive_prob", positive_prob[:,0])
                    tf.summary.histogram("negative_probs", negative_probs[:,0,:])
                
                
                #reg_loss = self.reg_weight_decay * sum(tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables() if not ("noreg" in tf_var.name or "Bias" in tf_var.name))
                reg_loss = tf.compat.v1.losses.get_regularization_loss()
                tf.compat.v1.summary.scalar("reg_loss", family='train', tensor=reg_loss)
                
                
                #XE loss
                xe_loss = tf.multiply(tf.math.log(positive_prob), loss_mask)

                
                #Averaging the loss by the number of masked items in the batch
                cosine_sim_loss = -tf.reduce_sum(xe_loss) / tf.reduce_sum(loss_mask) 
                tf.compat.v1.summary.scalar("cosine_sim_loss", family='train', tensor=cosine_sim_loss)
                
                self.total_loss = cosine_sim_loss + reg_loss

                #if mode == tf.estimator.ModeKeys.TRAIN:
                jobs_prob_masked = tf.multiply(jobs_prob, tf.expand_dims(loss_mask, -1), name='jobs_prob_masked_op')
            
            
            
            if mode == tf.estimator.ModeKeys.TRAIN:

                with tf.compat.v1.variable_scope('training'):
                    opt = tf.compat.v1.train.AdamOptimizer(
                        self.lr,
                        beta1=0.9,
                        beta2=0.999,
                        epsilon=1e-08)


                    #Necessary to run update ops for batch_norm, streaming metrics
                    update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)                
                    with tf.control_dependencies(update_ops):      
                        # Get the gradient pairs (Tensor, Variable)
                        grads = opt.compute_gradients(self.total_loss)
                        # Update the weights wrt to the gradient
                        # global_step + 1 就代表处理了一个batch
                        # global_step refers to the number of batches seen by the graph. 
                        self.train = opt.apply_gradients(grads, global_step=tf.compat.v1.train.get_global_step())
                        
                        if self.plot_histograms:
                            # Add histograms for trainable variables.
                            for grad, var in grads:
                                if grad is not None:
                                    tf.summary.histogram(var.op.name + '/gradients', grad)
                                    
    
    

    ### Session learning models
    #Good reference: https://github.com/tensorflow/magenta/blob/master/magenta/models/shared/events_rnn_graph.py
    def build_rnn(self, the_input, lengths, rnn_units=256, residual_connections=False):    
        with tf.compat.v1.variable_scope("RNN"):    
            fw_cells = []
            #bw_cells = []

            #Hint: Use tf.contrib.rnn.InputProjectionWrapper if the number of units between layers is different
            for i in range(self.rnn_num_layers):
                #cell = tf.nn.rnn_cell.GRUCell(rnn_units)  
                #cell = tf.nn.rnn_cell.LSTMCell(rnn_units, state_is_tuple=True)
                cell = tf.contrib.rnn.UGRNNCell(rnn_units) 

                if residual_connections:
                    cell = tf.contrib.rnn.ResidualWrapper(cell)
                    if i == 0: #or rnn_layer_sizes[i] != rnn_layer_sizes[i - 1]:
                        #cell = tf.contrib.rnn.InputProjectionWrapper(cell, rnn_layer_sizes[i])  
                        cell = tf.contrib.rnn.InputProjectionWrapper(cell, rnn_units)  
                        
                #cell = tf.contrib.rnn.AttentionCellWrapper(cell, attn_length=3, state_is_tuple=True)
                cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=self.dropout_keep_prob #, input_keep_prob=self.keep_prob
                                                     )
                fw_cells.append(cell)
            
            
            fw_stacked_cells = tf.contrib.rnn.MultiRNNCell(fw_cells, state_is_tuple=True)
            rnn_outputs, rnn_final_hidden_state_tuples = tf.nn.dynamic_rnn(fw_stacked_cells, the_input, dtype=tf.float32, sequence_length=lengths)
            if self.plot_histograms:
                tf.summary.histogram("rnn/outputs", rnn_outputs)   
            
            return rnn_outputs



    def build_pan(self, the_input, user_ids, lengths):
        with tf.compat.v1.variable_scope("PAN"):
            #sess = tf.Session()
            #print(sess.run([user_ids, tf.expand_dims(tf.convert_to_tensor(user_ids), 0)]))
            user_id_embeddings = self.users_id_embed(user_ids)
            user_id_embeddings = tf.squeeze(user_id_embeddings, axis=1)

            user_id_embedding_fc1 = tf.layers.dense(
                    user_id_embeddings,
                    100,
                    activation=tf.nn.leaky_relu, 
                    kernel_initializer=variance_scaling_initializer(),
                    kernel_regularizer=tf.contrib.layers.l2_regularizer(self.reg_l2_rate),
                    name="FC1")


            user_id_embedding_fc2 = tf.layers.dense(
                    user_id_embedding_fc1, 
                    self.pretrained_job_embedding_size,
                    activation=tf.nn.tanh, 
                    name='FC2', 
                    kernel_regularizer=tf.contrib.layers.l2_regularizer(self.reg_l2_rate))

            user_id_embeddings = tf.expand_dims(user_id_embedding_fc2, axis=-1)


            sess = tf.compat.v1.Session()

            score = tf.matmul(the_input, user_id_embeddings)
            score = tf.reduce_sum(score, axis=-1)
            score_tile = tf.tile(score, [1, tf.shape(score)[1]])
            score_reshape =  tf.reshape(score_tile, shape = [tf.shape(score)[0], -1, tf.shape(score)[1]])     
            score_diag = tf.linalg.band_part(score_reshape,-1,0)
            att = tf.nn.softmax(score_diag)
            #att = tf.layers.dropout(att, rate=1.0-self.dropout_keep_prob, training=self.is_training)
                
            outputs = tf.matmul(att, the_input)
            if self.plot_histograms:
                tf.summary.histogram("PAN/outputs", outputs)
                tf.summary.histogram("PAN/att", att)   
            return att, outputs
    
    
    ###### Get features #####
    
    def get_features(self, inputs, features_config, features_to_ignore):

        def cat_ohe(feature_name, size, inputs):
            return tf.one_hot(inputs[feature_name], size, name="{}_cat_one_hot".format(feature_name))
                
        def cat_embed(feature_name, size, inputs):
            with tf.compat.v1.variable_scope("{}_cat_embedding".format(feature_name), reuse=tf.compat.v1.AUTO_REUSE):        
                dim = get_embedding_size(size)
                embeddings = tf.compat.v1.get_variable("{}_embedding".format(feature_name), shape=[size, dim], regularizer=tf.contrib.layers.l2_regularizer(self.reg_l2_rate))
                lookup = tf.nn.embedding_lookup(embeddings, ids=inputs[feature_name])#, max_norm=1)
                return lookup
        
    
        with tf.compat.v1.variable_scope("features"):
            features_list = []
            for feature_name in features_config:
                #Ignores item_clicked and timestamp as user contextual features
                if feature_name in features_to_ignore:
                    continue

                if features_config[feature_name]['type'] == 'categorical':
                    size = features_config[feature_name]['cardinality']
                    if features_config[feature_name]['cardinality'] <= self.max_cardinality_for_ohe:
                        feature_op = cat_ohe(feature_name, size, inputs)
                    else:
                        feature_op = cat_embed(feature_name, size, inputs)
                elif features_config[feature_name]['type'] == 'numerical':
                    feature_op = tf.expand_dims(inputs[feature_name], -1)
                
                else:
                    raise Exception('Invalid feature type: {}'.format(feature_name))
                
                if self.plot_histograms:
                    tf.summary.histogram(feature_name, family='stats',values=feature_op)

                features_list.append(feature_op)
            
            if len(features_list) > 0:
                features_concat = tf.concat(features_list, axis=-1)
                return features_concat
            else:
                return None

            
    def get_job_features(self, job_ids, summary_suffix):
        with tf.compat.v1.variable_scope("job_features"):
            #Obtaining job features for specified job (e.g. clicked, negative samples)
            job_metadata_features_values = {}
            for feature_name in self.job_features_config:
                if feature_name not in JOB_REQ_FEATURES:
                    job_metadata_features_values[feature_name] = tf.gather(self.job_metadata[feature_name], job_ids)

            jobs_features_list = []
            if len(job_metadata_features_values) > 0:
                #Concatenating job contextual features
                job_metadata_features = self.get_features(job_metadata_features_values, features_config=self.job_features_config, features_to_ignore=JOB_REQ_FEATURES)
                
                #Adding job metadata attributes as input for the network
                jobs_features_list.append(job_metadata_features)
                
                if self.plot_histograms:
                    tf.summary.histogram('job_metadata_features/'+summary_suffix, family='stats', values=tf.boolean_mask(job_metadata_features, tf.cast(tf.sign(job_ids), tf.bool)))
            
            #If enabled, add Pre-trained Job Content Embeddings 
            if self.internal_features_config['job_content_embeddings']:
                jobs_embeddings_lookup = tf.nn.embedding_lookup(self.pretrained_job_content_embeddings, ids=job_ids)
                jobs_features_list.append(jobs_embeddings_lookup)

                if self.plot_histograms:
                    tf.summary.histogram('jobs_embeddings_lookup/'+summary_suffix, family='stats', values=tf.boolean_mask(jobs_embeddings_lookup, tf.cast(tf.sign(job_ids), tf.bool))) 
            
            
            #If enabled, adds trainable item embeddings
            if self.internal_features_config['job_clicked_embeddings']:
                job_clicked_interactions_embedding = self.jobs_id_embed(job_ids)
                jobs_features_list.append(job_clicked_interactions_embedding)

                if self.plot_histograms:
                    tf.summary.histogram('job_clicked_interactions_embedding/'+summary_suffix, family='stats', values=tf.boolean_mask(job_clicked_interactions_embedding, tf.cast(tf.sign(job_ids), tf.bool)))

           
            jobs_features_concat = tf.concat(jobs_features_list, axis=-1)
            
            return jobs_features_concat


    def jobs_id_embed(self, job_ids):
        #with tf.device('/cpu:0'):
        with tf.compat.v1.variable_scope("job_id_embedding", reuse=tf.compat.v1.AUTO_REUSE):
            size = self.job_vocab_size
            dim = get_embedding_size(size)
            embeddings = tf.compat.v1.get_variable("job_id_embedding", shape=[size, dim], regularizer=tf.contrib.layers.l2_regularizer(self.reg_l2_rate))
            lookup = tf.nn.embedding_lookup(embeddings, ids=job_ids)#, max_norm=1)
            return lookup

    def users_id_embed(self, user_ids):
        with tf.compat.v1.variable_scope("user_id_embedding", reuse=tf.compat.v1.AUTO_REUSE):
            size = self.user_features_config['UserID']['cardinality']
            dim = get_embedding_size(size)
            embeddings = tf.compat.v1.get_variable("user_id_embedding", shape=[size, dim], regularizer=tf.contrib.layers.l2_regularizer(self.reg_l2_rate))
            lookup = tf.nn.embedding_lookup(embeddings, ids=user_ids)#, max_norm=1)
            return lookup


    
    def scale_center_features(self, job_features, begin_norm_axis=2):
        with tf.compat.v1.variable_scope("input_features_center_scale", reuse=tf.compat.v1.AUTO_REUSE):
            gamma = tf.compat.v1.get_variable("gamma_scale", 
                                    shape=[job_features.get_shape()[-1]], 
                                    initializer=tf.ones_initializer(),
                                    regularizer=tf.contrib.layers.l2_regularizer(self.reg_l2_rate))
            beta = tf.compat.v1.get_variable("beta_center", 
                                   shape=[job_features.get_shape()[-1]], 
                                   initializer=tf.zeros_initializer(),
                                   regularizer=tf.contrib.layers.l2_regularizer(self.reg_l2_rate))

            if self.plot_histograms:
                tf.summary.histogram('input_features_gamma_scale', family='stats', values=gamma)
                tf.summary.histogram('input_features_beta_center', family='stats', values=beta)


            job_features_centered_scaled = (job_features * gamma) + beta

        return job_features_centered_scaled
    
    


    
    ########### Negative Samples ##########
    
    def get_sample_from_recently_clicked_jobs_buffer(self, sample_size):
        #pop_recent_jobs_buffer_masked_by_state = {}
        with tf.compat.v1.variable_scope("neg_samples_buffer"):
            pop_recent_jobs_buffer_masked = tf.boolean_mask(self.pop_recent_jobs_buffer, tf.cast(tf.sign(self.pop_recent_jobs_buffer), tf.bool))             


            tf.compat.v1.summary.scalar('clicked_jobs_on_buffer', family='stats', tensor=tf.shape(pop_recent_jobs_buffer_masked)[0])

            recent_jobs_unique_sample = tf.random.shuffle(pop_recent_jobs_buffer_masked)
            
            #Samples K jobs from recent jobs
            sample_recently_clicked_jobs = recent_jobs_unique_sample[:sample_size]
            return sample_recently_clicked_jobs


    def get_batch_negative_samples(self, all_clicked_jobs, user_cities, user_states, additional_samples, num_negative_samples, first_sampling_multiplying_factor=2):
        with tf.compat.v1.variable_scope("neg_samples_batch"):
            #current_batch_size, batch_max_session_length = tf.shape(item_clicked)[0], tf.shape(item_clicked)[1] 

            # all_clicked_jobs 类似 [[1, 2, 3, 7, 8, 9], [4, 5, 6, 10, 11, 12]]
            # batch_jobs = [ 1,  2,  3,  7,  8,  9,  4,  5,  6, 10, 11, 12]
            batch_jobs = tf.reshape(all_clicked_jobs, [-1])
            batch_jobs_states = tf.reshape(self.batch_jobstate, [-1])

            #Removing padded (zeroed) items
            batch_jobs_non_zero = tf.boolean_mask(batch_jobs, tf.cast(tf.sign(batch_jobs), dtype=tf.bool))
            batch_jobs_states_non_zero = tf.boolean_mask(batch_jobs_states, tf.cast(tf.sign(batch_jobs_states), dtype=tf.bool))

            candidate_neg_jobs = tf.concat([batch_jobs_non_zero, additional_samples], axis=0)  
            additional_samples_state = tf.gather(self.job_metadata['JobState'], additional_samples)  
            candidate_neg_jobs_state = tf.concat([batch_jobs_states_non_zero, additional_samples_state], axis=0) 


            # candidate_neg_jobs_shuffled_by_state shape=(?, ?)

            candidate_neg_jobs_shuffled_by_state = tf.map_fn(lambda user_state: self.get_candidate_neg_jobs_by_state(user_state, candidate_neg_jobs, candidate_neg_jobs_state, num_negative_samples, first_sampling_multiplying_factor=2), user_states)
            

            batch_negative_jobs = self.get_negative_samples(all_clicked_jobs, user_states, candidate_neg_jobs_shuffled_by_state, num_negative_samples)
            return batch_negative_jobs

    # Step 2
    def get_candidate_neg_jobs_by_state(self, user_state, batch_jobs_non_zero, batch_jobs_states_non_zero, num_negative_samples, first_sampling_multiplying_factor=2):
        with tf.compat.v1.variable_scope("candidate_neg"):
            # batch_jobs_non_zero shape=(?,)
            # user_state shape=(1,)
            # tf.equal(batch_jobs_states_non_zero, user_state) shape=(?,)
            # tf.where(tf.equal(batch_jobs_states_non_zero, user_state)) shape=(?, 1)
            negative_samples = tf.gather_nd(batch_jobs_non_zero, tf.where(tf.equal(batch_jobs_states_non_zero, user_state)))
            negative_samples = tf.concat([negative_samples, batch_jobs_non_zero], axis = 0)
            negative_samples = negative_samples[:(num_negative_samples*first_sampling_multiplying_factor)]
            # shape=(?,)

            return negative_samples


    def get_negative_samples(self, all_clicked_jobs, user_states, candidate_samples, num_neg_samples):  
        with tf.compat.v1.variable_scope("negative_samples"):
            # user_states shape=(?, 1)
            # all_clicked_jobs shape=(?, ?)
            # indices shape=(?,)
            indices = tf.range(tf.shape(user_states)[0])
            indices = tf.cast(indices, tf.int64)
            shuffled_neg_samples = tf.map_fn(lambda ii: self.get_neg_jobs_session(ii, all_clicked_jobs, candidate_samples, num_neg_samples), indices)

            return shuffled_neg_samples


    def get_neg_jobs_session(self, index, all_clicked_jobs, all_candidate_samples, num_neg_samples):
        #Ignoring negative samples clicked within the session (keeps the order and repetition of candidate_samples)
        # session_job_ids shape=(?,)
        # candidate_samples shape=(?,)
        session_job_ids = all_clicked_jobs[index]
        candidate_samples = all_candidate_samples[index]
        # valid_samples_session shape=(?,)
        valid_samples_session, _ = tf.setdiff1d(candidate_samples, session_job_ids, index_dtype=tf.int64)
        #Generating a random list of negative samples for each click (with no repetition)
        # session_clicks_neg_items shape=(?, ?)
        session_clicks_neg_jobs = tf.map_fn(lambda click_id: tf.cond(tf.equal(click_id, tf.constant(0, tf.int64)), lambda: tf.zeros(num_neg_samples, tf.int64), lambda: self.get_neg_jobs_click(valid_samples_session, num_neg_samples)), session_job_ids)                                                     
        return session_clicks_neg_jobs
   

    
    def get_neg_jobs_click(self, valid_samples_session, num_neg_samples):
        #Shuffles neg. samples for each click
        valid_samples_shuffled = tf.random.shuffle(valid_samples_session)
        samples_unique_vals, samples_unique_idx = tf.unique(valid_samples_shuffled)

        #Returning first N unique items (to avoid repetition)
        first_unique_items = tf.math.unsorted_segment_min(data=valid_samples_shuffled,
                                                     segment_ids=samples_unique_idx,
                                                     num_segments=tf.shape(samples_unique_vals)[0])[:num_neg_samples]

        #Padding if necessary to keep the number of neg samples constant (ex: first batch)
        first_unique_items_padded_if_needed = tf.concat([first_unique_items, tf.zeros(num_neg_samples-tf.shape(first_unique_items)[0], tf.int64)], axis=0)

        return first_unique_items_padded_if_needed                            

    

    ##### Prediction #####
    def rank_jobs_by_predicted_prob(self, job_ids, jobs_prob):
        with tf.compat.v1.variable_scope("predicted_jobs"):
            #Ranking job ids by their predicted probabilities
            jobs_top_prob = tf.nn.top_k(jobs_prob,  k=tf.shape(jobs_prob)[2])
            jobs_top_prob_indexes = jobs_top_prob.indices
            predicted_job_probs = jobs_top_prob.values

            jobs_top_prob_indexes_idx = tf.contrib.layers.dense_to_sparse(jobs_top_prob_indexes, eos_token=-1).indices
            jobs_top_prob_indexes_val = tf.gather_nd(jobs_top_prob_indexes, jobs_top_prob_indexes_idx)
            #Takes the first two columns of the index and use sorted indices as the last column
            jobs_top_prob_reordered_indexes = tf.concat([jobs_top_prob_indexes_idx[:,:2], tf.expand_dims(tf.cast(jobs_top_prob_indexes_val, tf.int64), 1)], 1)
            predicted_job_ids = tf.reshape(tf.gather_nd(job_ids, jobs_top_prob_reordered_indexes), tf.shape(job_ids))
            return predicted_job_ids, predicted_job_probs
    
    
    
    def define_mrr_metric(self, predicted_job_ids, next_job_label_expanded, topk):
        with tf.compat.v1.variable_scope("mrr"):
            reciprocal_ranks = tf.div(tf.constant(1.0),
                                      tf.cast(tf.constant(1, tf.int64) + tf.where(tf.logical_and(tf.equal(next_job_label_expanded, predicted_job_ids[:,:,:topk]), tf.expand_dims(self.job_clicked_mask, -1)
                                                                                                 #Apply mask to sessions with padded items
                                                                                                ))[:,2],tf.float32)) 


            batch_valid_labels_count = tf.reduce_sum(tf.to_int32(self.job_clicked_mask))
            batch_labels_not_found_in_topk = batch_valid_labels_count - tf.size(reciprocal_ranks)


            #Completing with items for which the label was not in the preds (because tf.where() do not return indexes in this case), 
            #so that mean is consistent
            reciprocal_ranks = tf.concat([reciprocal_ranks, tf.zeros(batch_labels_not_found_in_topk)], axis=0)
            
            mrr, mrr_update_op = tf.compat.v1.metrics.mean(values=reciprocal_ranks,name='mrr_at_n')              

            return mrr, mrr_update_op
        
    
    def define_ndcg_metric(self, predicted_job_ids, next_job_label_expanded, topk):
        with tf.compat.v1.variable_scope("ndcg"):
            #Computing NDCG
            predicted_correct = tf.to_int32(tf.equal(predicted_job_ids, next_job_label_expanded))
            ndcg_predicted = tf_ndcg_at_k(predicted_correct, topk)
            
            #Combining masks of padding items and NDCG zeroed values (because the correct value is not in the top n)
            ndcg_mask = tf.to_float(self.job_clicked_mask)
            
            ndcg_mean, ndcg_mean_update_op = tf.compat.v1.metrics.mean(values=ndcg_predicted, weights=ndcg_mask, name='ndcg_at_n')              

            return ndcg_mean, ndcg_mean_update_op



            
    def define_eval_metrics(self, next_job_label, predicted_job_ids):
        with tf.compat.v1.variable_scope("evaluation_metrics"):


            next_job_label_expanded = tf.expand_dims(next_job_label, -1)
            print('next_job_label_expanded', next_job_label_expanded)
            
            #Computing Recall@N
            self.recall_at_n, self.recall_at_n_update_op = tf.contrib.metrics.sparse_recall_at_top_k(
                labels=next_job_label_expanded,
                top_k_predictions=predicted_job_ids[:,:,:self.metrics_top_n],
                weights=tf.to_float(self.job_clicked_mask), 
                name='hitrate_at_n')
            
            #Computing MRR@N
            self.mrr, self.mrr_update_op = self.define_mrr_metric(
                predicted_job_ids,
                next_job_label_expanded,
                topk=self.metrics_top_n)
            
            
            #Computing NDCG@N
            self.ndcg_at_n_mean, self.ndcg_at_n_mean_update_op = self.define_ndcg_metric(
                predicted_job_ids,
                next_job_label_expanded,
                topk=self.metrics_top_n)