import numpy as np
from scipy.linalg import toeplitz
from sklearn.linear_model import Ridge, RidgeCV, LinearRegression
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

class receptive_field_models:
    
    def __init__(self):
        self.dm = None
        
    def design_matrix(self, stim, kshape): 
        """
        Create a block toeplitz of a stimulus with 2D frames over time.
        Adapted from: https://github.com/melizalab/comp-neurosci/blob/master/4-Receptive-Fields.ipynb
        """
        
        # preliminary checks
        if len(stim.shape) == 3:
            # reshape stim to reflect nchannels x ntimesteps
            sdim0, sdim1, st = stim.shape
            stim = stim.reshape(sdim0*sdim1, st)
        else:
            pass
        
        if len(kshape) == 3:    
            # get nchannels x ntimesteps of the desired kernel
            kshape = (kshape[0]*kshape[1], kshape[2])
        else:
            pass
        
        # make a block toeplitz of the stimulus
        assert stim.shape[0] == kshape[0], "Number of channels doesn't match between stim and RF"

        # M is the number of channels, N is the number of timesteps
        M, N = stim.shape

        # initialize a list to hold the toeplitz matricies for each channel
        toeps = []

        # for each channel
        for i in range(M):

            first = np.zeros(kshape[1]) # make a padding array with the same length as the kernel timespan

            # create the toeplitz matrix for the stim at channel i
            Xi = toeplitz(stim[i], first)

            # append to the holding list
            toeps.append(Xi)

            del Xi

        # column stacking a tensor (cube matrix) returns a matrix where 
        # the dim 2 is stacked in dim 1 and dim 0 is unaffected
        X = np.column_stack(toeps)
        assert X.shape[1] == kshape[0]*kshape[1], "design matrix does not have the right shape"

        return X
    
    def clear_cached_dm(self):
        self.dm = None
        
    def map_rf(self, x, Y, kshape, reg_type = 'ridge', transform = None, **kwargs):
        
        # set up argument constraints       
        models = {
            'linearregression': LinearRegression,
            'ridge': Ridge, 
            'ridgecv': RidgeCV,
        }
        
        transforms = {
            None: None,
            'pca': self.pca_helper,
        }
        
        kwarg_constraints = [
            'alphas',
            'n_components',
        ]
        
        if reg_type not in list(models.keys()):
            return f'Error: Not a valid regression method. Valid regression methods include: {list(models.keys())} '
        
        if transform not in list(transforms.keys()):
            return f'Error: Not a valid transformation method. Valid transformation methods include: {list(transforms.keys())}'
        
        for kwarg in list(kwargs.keys()):
            if kwarg not in kwarg_constraints:
                return f'Error: {kwarg} is not a valid key word argument. Valid key word arguments include: {kwarg_constraints}'
        
        # create the design matrix
        if x == 'cache':
            X = self.dm
        else:
            X = self.design_matrix(x, kshape)
        
        # apply transformation
        if transform != None:
            k = transforms[transform](
                X, Y,
                reg_type,
                models[reg_type],
                kwargs
            )
            
        else: # pick a model
            # if model involves regularization
            try:
                alphas = kwargs['alphas']
                model = models[reg_type](alphas)
            except KeyError: # otherwise run normally
                model = models[reg_type]()
                
            # fit the model
            model.fit(X, Y)
            
            # get the RF
            k = model.coef_
        
        # reshape the RF
        rf = k.reshape(kshape).T
        
        return rf
            
    def pca_helper(self, X, Y, reg_type, model, kwargs):
        # if regression has a regularization method
        try:
            alphas = kwargs['alphas']
            pcr = make_pipeline(StandardScaler(), 
                                PCA(n_components=kwargs['n_components']), 
                                model(alphas))
            
        # otherwise run the model normally
        except KeyError:
            pcr = make_pipeline(StandardScaler(), 
                                PCA(n_components=kwargs['n_components']), 
                                model())
        # fit the model
        pcr.fit(X, Y)
        
        # get the transformed coeficients
        transformed_coef = pcr.named_steps[reg_type].coef_
        
        # get the PCA components
        components = pcr.named_steps["pca"].components_.T
        
        # transform to the correct number of coeficients
        coeficients = components @ transformed_coef
        
        return coeficients
            
    def encode(self, stim, k, nonlinearity = 'exp', y_scale = 1.0, y_offset = 0):
        # preliminary checks
        if len(stim.shape) == 3:
            # reshape stim to reflect (nchannels, ntimesteps)
            stim = stim.reshape(stim.shape[0]*stim.shape[1], stim.shape[2])
        else:
            pass
        
        if len(k.shape) == 3:
            # reshape k to reflect (nchannels, ntimesteps)
            k = k.reshape(k.shape[0]*k.shape[1], k.shape[2])
        else:
            pass
        
        assert stim.shape[0] == k.shape[0], "Number of channels doesn't match between stim and RF"
        
        # prepare design matrix & RF
        self.dm = self.design_matrix(stim, k.shape)
        X, K = (self.dm, k.flatten())
        
        # dot product of X and K
        Y = np.dot(X, K)
        
        # apply scale and offset
        ypred = (Y/y_scale)-y_offset
        
        # apply a nonlinearity
        nonlin_map = {
            'exp': np.exp(ypred-(ypred.max()-1))
        }
        
        if nonlinearity not in nonlin_map.keys():
            print(f"nonlinearity parameter must be one of the following: {list(nonlin_map.keys())}, defaulted to exponentiation.")
            nonlinearity = 'exp'
        
        return ypred, nonlin_map[nonlinearity]