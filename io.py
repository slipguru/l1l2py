import os
from ConfigParser import SafeConfigParser, NoOptionError, NoSectionError

import scipy.io as sio

class Configuration(object):
    """ Configuration file object
    """    
    def __init__(self, conf_file):
        self._conf = SafeConfigParser()
        self._path = os.path.abspath(conf_file)
        if not self._conf.read(self._path):
            raise RuntimeError('not valid configuration file')
            
        #Lazy initializations
        self._expressions = None
        self._expressions_path = None
        self._labels = None
        self._labels_path = None
        self._options = None
        self._tau_range = None
        self._lambda_range = None
        self._mu_range = None

    @property
    def path(self):
        return self._path
    
    @property
    def experiment_type(self):
        return self._conf.get('experiment', 'type')
    
    @property
    def experiment_types(self):
        return ('classification', 'regression')
        
    @property
    def expressions_path(self):
        if not self._expressions_path: self._set_paths()
        return self._expressions_path
    
    @property
    def labels_path(self):
        if not self._labels_path: self._set_paths()
        return self._labels_path
    
    def _set_paths(self):
        config_path = os.path.split(self._path)[0]
        self._expressions_path = os.path.join(config_path,
                                       self._conf.get('expressions', 'path'))
        self._labels_path = os.path.join(config_path,
                                       self._conf.get('labels', 'path'))
           
    @property
    def expressions(self):
        if self._expressions is None:
            expressions_type = self._conf.get('expressions', 'type')
            if expressions_type == 'matlab':
                if self.expressions_path == self.labels_path:
                    self._expressions, self._labels = \
                                self._get_data(self._conf.get('expressions',
                                                              'name'),
                                               self._conf.get('labels',
                                                              'name'))
                else:
                    self._expressions = \
                                self._get_data(self._conf.get('expressions',
                                                              'name'))
        return self._expressions
    
    @property
    def labels(self):
        if self._labels is None:
            labels_type = self._conf.get('labels', 'type')
            if labels_type == 'matlab':
                if self.expressions_path == self.labels_path:
                    self._expressions, self._labels = \
                                self._get_data(self._conf.get('expressions',
                                                              'name'),
                                               self._conf.get('labels',
                                                              'name'))
                else:
                    self._labels = self._get_data(self._conf.get('labels',
                                                                 'name'))
        return self._labels
    
    #TODO: think better data reading!
    
    def _get_data(self, *names):
        raw_data = sio.loadmat(self.expressions_path, struct_as_record=False)
        
        data = list()
        for n in names:
            data.append(raw_data[n])
        
        return data if len(data) > 1 else data[0]
        
    @property
    def data_types(self):
        return ('matlab', 'csv')
        
    @property
    def tau_range_type(self):
        return self._conf.get('parameters', 'tau-range-type')
        
    @property
    def lambda_range_type(self):
        return self._conf.get('parameters', 'lambda-range-type')
        
    @property
    def mu_range_type(self):
        return self._conf.get('parameters', 'mu-range-type')
        
    @property
    def range_types(self):
        return ('linear', 'geometric')
    
    @property
    def tau_range(self):
        if self._tau_range is None:
            self._tau_range = self._get_range_values('tau',
                                                     self.tau_range_type)
        return self._tau_range
        
    @property
    def lambda_range(self):
        if self._lambda_range is None:
            self._lambda_range = self._get_range_values('lambda',
                                                        self.lambda_range_type)
        return self._lambda_range
        
    @property
    def mu_range(self):
        if self._mu_range is None:
            self._mu_range = self._get_range_values('mu', self.mu_range_type)
        return self._mu_range
        
    def _get_range_values(self, param, type):
        import tools
        min = self._conf.getfloat('parameters', '%s-min' % param)
        max = self._conf.getfloat('parameters', '%s-max' % param)
        num = self._conf.getint('parameters', '%s-number' % param)
        return tools.parameter_range(type, min, max, num)
    
    @property
    def raw_options(self):
        if not self._options:
            sections = self._conf.sections()
            self._options = dict()
            for s in sections:
                for o, v in self._conf.items(s):
                    self._options['%s_%s' % (s, o)] = v
        return self._options
    
    @property
    def external_k(self):
        return self._conf.getint('parameters', 'external-k')
    
    @property
    def internal_k(self):
        return self._conf.getint('parameters', 'internal-k')
        
    @property
    def split_index(self):
        return self._conf.getint('parameters', 'split-index')
    
    def __str__(self):
        relative_path = os.path.relpath(self.path)
        width = 55
        just_l, just_r = (width/2)-7, (width/2)+5
        
        separator = '+' + '-'*width + '+\n'
        
        header = separator + \
                 '|' + relative_path.center(width) + '|\n' + \
                 separator
        content = '\n'.join(
            '| ' + k.rjust(just_r) + ': ' + self.raw_options[k].ljust(just_l) + '|'
                            for k in sorted(self.raw_options.keys()) )
        footer = '\n' + separator
        
        return ''.join((header, content, footer))


