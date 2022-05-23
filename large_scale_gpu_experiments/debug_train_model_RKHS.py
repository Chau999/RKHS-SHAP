from utils import *
# import pykeops
# pykeops.clean_pykeops()

import os
# pykeops.verbose = True
# pykeops.build_type = 'Debug'

# Clean up the already compiled files
# pykeops.clean_pykeops()
#
# pykeops.test_torch_bindings()

if __name__ == '__main__':
    train_params={
        'dataset':'website_data',
        'fold':0,
        'epochs':100,
        'patience':5,
        'bs':1000,
        'double_up': False,
        'm_factor': 5.,
        'seed': 42,
        'folds': 5,
    }
    for ds in [['census',1.0],['LOL',1.0]]:
        for f in [0]:
            train_params['dataset']=ds[0]
            train_params['model_string']='SGD_base'
            train_params['fold']=f
            train_params['m_factor']=ds[1]
            c= train_krr_simple(train_params=train_params)
            c.train_model()