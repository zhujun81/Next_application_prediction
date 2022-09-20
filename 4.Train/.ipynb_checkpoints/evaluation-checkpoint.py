import tensorflow as tf

from utils import log_base


def update_metrics(preds, labels, clicked_jobs, streaming_metrics, recommender=''):
    for metric in streaming_metrics:
        metric.add(preds, labels)



def compute_metrics_results(streaming_metrics, recommender=''):
    results = {}
    for metric in streaming_metrics:
        result = metric.result()
        results['{}_{}'.format(metric.name, recommender)] = result

    return results


def tf_ndcg_at_k(r, k):
    def _tf_dcg_at_k(r, k):
        last_dim_size = tf.minimum(k, tf.shape(r)[-1])

        input_rank = tf.rank(r)
        input_shape = tf.shape(r)    
        slice_begin = tf.zeros([input_rank], dtype=tf.int32)
        slice_size = tf.concat([input_shape[:-1], [last_dim_size]], axis=0)
        r = tf.slice(tf.to_float(r),
                     begin=slice_begin,
                     size=slice_size)

        last_dim_size = tf.shape(r)[-1]

        dcg = tf.reduce_sum(tf.subtract(tf.pow(2., r), 1) / log_base(tf.range(2, last_dim_size + 2), 2.), axis=-1)

        return dcg    
    
    sorted_values, sorted_idx = tf.nn.top_k(r, k=tf.shape(r)[-1])
    idcg = _tf_dcg_at_k(sorted_values, k)
    
    ndcg = _tf_dcg_at_k(r, k) / idcg
    #Filling up nans (due to zeroed IDCG) with zeros
    ndcg = tf.where(tf.is_nan(ndcg), tf.zeros_like(ndcg), ndcg)

    return ndcg




