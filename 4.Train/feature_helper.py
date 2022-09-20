def get_job_features_config(job_label_encoders):
    job_features_config = {
    #Required fields
    'JobID': {'type': 'categorical', 'dtype': 'int'},
    'StartDate': {'type': 'numerical', 'dtype': 'int'},
    'EndDate': {'type': 'numerical', 'dtype': 'int'},

    #Additional metadata fields
    'JobCity': {'type': 'categorical', 'dtype': 'int'},
    'JobState': {'type': 'categorical', 'dtype': 'int'},  
    'JobCountry': {'type': 'categorical', 'dtype': 'int'},
    }

    #Adding cardinality to categorical features
    for feature_name in job_features_config:
        if feature_name in job_label_encoders and job_features_config[feature_name]['type'] == 'categorical':
            job_features_config[feature_name]['cardinality'] = len(job_label_encoders[feature_name])

    print('Job Features: {}'.format(job_features_config))       
    return job_features_config



def get_user_features_config(user_label_encoders):
    user_features_config = {
    #Required fields
    'UserID': {'type': 'categorical', 'dtype': 'int'},

    #Additional metadata fields
    'UserCity': {'type': 'categorical', 'dtype': 'int'},
    'UserState': {'type': 'categorical', 'dtype': 'int'},  
    'UserCountry': {'type': 'categorical', 'dtype': 'int'},
    'UserDegree': {'type': 'categorical', 'dtype': 'int'},  
    'UserMajor': {'type': 'categorical', 'dtype': 'int'},
    

    }

    #Adding cardinality to categorical features
    for feature_name in user_features_config:
        if feature_name in user_label_encoders and user_features_config[feature_name]['type'] == 'categorical':
            user_features_config[feature_name]['cardinality'] = len(user_label_encoders[feature_name])

    print('User Features: {}'.format(user_features_config))       
    return user_features_config


def get_session_features_config(job_label_encoders, user_label_encoders):
    session_features_config = {
    'single_features': {
    #Control features
    'SessionID': {'type': 'numerical', 'dtype': 'int'},
    'UserID': {'type': 'numerical', 'dtype': 'int'},
    'SessionSize': {'type': 'numerical', 'dtype': 'int'},
    'SessionStart': {'type': 'numerical', 'dtype': 'int'},            
    },

    'sequence_features': {
    #Required sequence features
    'ApplicationDate': {'type': 'numerical', 'dtype': 'int'},
    'Job_clicked': {'type': 'categorical', 'dtype': 'int'},

    #Location
    'JobCity': {'type': 'categorical', 'dtype': 'int'},
    'JobState': {'type': 'categorical', 'dtype': 'int'},
    'JobCountry': {'type': 'categorical', 'dtype': 'int'}, 
    'UserCity': {'type': 'categorical', 'dtype': 'int'}, 
    'UserState': {'type': 'categorical', 'dtype': 'int'},
    'UserCountry': {'type': 'categorical', 'dtype': 'int'}, 
    #Education
    'UserDegree': {'type': 'categorical', 'dtype': 'int'}, 
    'UserMajor': {'type': 'categorical', 'dtype': 'int'}, 
    }
    } 
    feature_groups = {
        'Location': ['JobCity', 'JobState', 'Country', 'UserCity', 'UserState', 'UserCountry'],
        'Education': ['UserDegree', 'UserMajor'],
    }
    
    #Adding cardinality to categorical features
    for feature_groups_key in session_features_config:
        features_group_config = session_features_config[feature_groups_key]
        for feature_name in features_group_config:
            if feature_name in user_label_encoders and features_group_config[feature_name]['type'] == 'categorical':
                features_group_config[feature_name]['cardinality'] = len(user_label_encoders[feature_name])
            if feature_name in job_label_encoders and features_group_config[feature_name]['type'] == 'categorical':
                features_group_config[feature_name]['cardinality'] = len(job_label_encoders[feature_name])
    
    
    print('Session Features: {}'.format(session_features_config))
    return session_features_config

