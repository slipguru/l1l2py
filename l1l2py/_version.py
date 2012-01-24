def _get_version(version, status=None):
    if status:
       return '%s-%s' % (version, status)
    return version

# major number for main changes
# minor number for new features
# release number for bug fixes and minor updates
# status = {'alpha', 'beta', None}
version = _get_version('1.0.5', status=None)
