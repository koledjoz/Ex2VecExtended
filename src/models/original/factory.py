from model import Ex2VecOriginal


def generate_model(config, type='original'):
    if type == 'original':
        return Ex2VecOriginal(config)
    else:
        raise ValueError(f'No model with the type {type} found.')