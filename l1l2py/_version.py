def _get_version(version, status=None):
    if status:
        import datetime
        now = datetime.datetime.now()
        date = '.'.join('%02d'%x for x in (now.year, now.month, now.day))
        return '%s%s-%s' % (version, status, date)
    return version

# major number for main changes
# minor number for new features
# release number for bug fixes and minor updates
# status = {'alpha', 'beta', None}
version = _get_version('1.1.0', status='alpha')
