def define_Dataset(dataset_opt):
    dataset_type = dataset_opt['dataset_type'].lower()
    if dataset_type in ['jpeg']:
        from data.dataset_jpeg import DatasetJPEG as D

    elif dataset_type in ['jpeggray']:
        from data.dataset_jpeggray import DatasetJPEG as D

    elif dataset_type in ['jpeggraydouble']:
        from data.dataset_jpeggraydouble import DatasetJPEG as D

    else:
        raise NotImplementedError('Dataset [{:s}] is not found.'.format(dataset_type))

    dataset = D(dataset_opt)
    print('Dataset [{:s} - {:s}] is created.'.format(dataset.__class__.__name__, dataset_opt['name']))
    return dataset
