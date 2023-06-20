
class LoggingModule:
    log_entry = {}
    enabled = False

    @staticmethod
    def log(name, value):
        if LoggingModule.enabled:
            if name in LoggingModule.log_entry:
                if not isinstance(LoggingModule.log_entry[name], list):
                    LoggingModule.log_entry[name] = [LoggingModule.log_entry[name]]
                LoggingModule.log_entry[name].append(value)
            else:
                LoggingModule.log_entry[name] = value

    @staticmethod
    def get_log():
        log = {}
        for k, v in LoggingModule.log_entry.items():
            if isinstance(v, list):
                for i, v_ in enumerate(v):
                    log[k+'/%d' % i] = v_
            else:
                log[k] = v
        LoggingModule.clear_log()
        return log

    @staticmethod
    def clear_log():
        LoggingModule.log_entry = {}






