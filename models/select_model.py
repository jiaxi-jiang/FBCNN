
"""
# --------------------------------------------
# define training model
# --------------------------------------------
"""


def define_Model(opt):
    model = opt['model']      # one input: L

    if model == 'fbcnn':
        from models.model_fbcnn import ModelFBCNN as M

    elif model == 'gan':     # one input: L
        from models.model_gan import ModelGAN as M

    else:
        raise NotImplementedError('Model [{:s}] is not defined.'.format(model))

    m = M(opt)

    print('Training model [{:s}] is created.'.format(m.__class__.__name__))
    return m
