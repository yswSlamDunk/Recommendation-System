import os
import pickle


def recommendation(user_name, user_in_data, K):

    base_path = os.path.join('saved', user_name, 'models')
    recent_folder = sorted(os.listdir(base_path))[-1]
    recent_model = pickle.load(os.path.join(
        base_path, recent_folder, 'model.pickle'))
    recommendation_item = recent_model.recommendation(user_in_data, K)

    return recommendation_item
