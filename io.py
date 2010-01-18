class Configuration(object):
    """ Configuration file object

    Data type???

    Example
    --------

    >>> c = Configuration('example_data/example.cfg')
    >>> c['experiment_type']
    'regression'
    >>> c['parameters_internal-k']
    '2'
    >>>
    """

    def __init__(self, file_path):
        from ConfigParser import SafeConfigParser
        self._conf = SafeConfigParser()

        if not self._conf.read(file_path):
            raise RuntimeError('incorrect configuration file') # WHEN!?

        self._path = file_path

    @property
    def path(self):
        return self._path

    def __getitem__(self, key):
        from ConfigParser import NoOptionError, NoSectionError
        section, option = key.split('_', 1) # max 1 split
        try:
            return self._conf.get(section, option)
        except (NoOptionError, NoSectionError), err:
            raise KeyError(err.message)

    def keys(self):
        return ['_'.join((s, o)) for s in self._conf.sections()
                                 for o in self._conf.options(s)]

    def data_matrices(self):
        """ To make lazy!! """
        ## if is a matlab file
        from scipy.io import loadmat
        from os import path

        # Configuration file directory is the root directory
        config_path = path.split(self.path)[0]

        tmp = loadmat(path.join(config_path, self['data_path']),
                      struct_as_record=True)
        expressions = tmp[self['data_name']]

        if self['data_path'] != self['labels_path']:
            tmp = loadmat(self['labels_path'], struct_as_record=True)
        labels = tmp[self['labels_name']]

        return expressions, labels

    def __str__(self):
        width = 55
        just_l, just_r = (width/2)-7, (width/2)+5
        separator = '+' + '-'*width + '+\n'

        header = separator + \
                 '|' + 'Input'.center(width) + '|\n' + \
                 separator
        content = '\n'.join(
            '| ' + k.rjust(just_r) + ': ' + self[k].ljust(just_l) + '|'
                            for k in sorted(self.keys()) )
        footer = '\n' + separator

        return ''.join((header, content, footer))

from nose import tools as t
class TestConfiguration(object):
    def setup(self):
        self.conf = Configuration('example_data/example.cfg')

    @t.raises(RuntimeError)
    def test_raise(self):
        Configuration('example_data/foo.cfg')

    def test_read(self):
        expected = self.conf._conf.get('experiment', 'type')
        t.assert_equals(expected, self.conf['experiment_type'])

    @t.raises(KeyError)
    def test_bad_key(self):
        self.conf['experiment_foo']

    @t.raises(KeyError)
    def test_bad_key2(self):
        self.conf['foo_type']

    def test_keys(self):
        t.assert_true('experiment_type' in self.conf.keys())

#def read_configuration_file(file_path):
#
#    # To generalize!!
#    input = {'experiment_type': config.get('experiment', 'type'),
#             'data_path': config.get('data', 'path'),
#             'data_name': config.get('data', 'name'),
#
#             'labels_path': config.get('labels', 'path'),
#             'labels_name': config.get('labels', 'name'),
#             'labels_type': config.get('labels', 'type'),
#
#             'result_path': config.get('output', 'result'),
#
#             'tau_min': config.getfloat('parameters', 'tau-min'),
#             'tau_max': config.getfloat('parameters', 'tau-max'),
#             'lambda_min': config.getfloat('parameters', 'lambda-min'),
#             'lambda_max': config.getfloat('parameters', 'lambda-max'),
#             'mu_min': config.getfloat('parameters', 'mu-min'),
#             'mu_max': config.getfloat('parameters', 'mu-max'),
#
#             'split_idx': config.getint('parameters', 'split-index'),
#             'external_k': config.getint('parameters', 'external-k'),
#             'internal_k': config.getint('parameters', 'internal-k'),
#            }
#
#    return input