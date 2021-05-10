""" Contains implementation of Config class. """


class Config(dict):

    def __new__(cls, value=None, **kwargs):
        if isinstance(value, dict):
            return cls.parse(value) @ cls.parse(kwargs)
        return super(Config, cls).__new__(cls)

    def __init__(self, value=None, **kwargs):
        super().__init__(**kwargs)

    @classmethod
    def check(cls, config, key=None,
              value=None, rules=None):
        if rules is not None and not isinstance(rules, (list, tuple)):
            rules = [rules]

        if not (key is None or key in config):
            raise ValueError("Key '{}' not present in config".format(key))
        elif value is not None:
            if config[key] != value:
                raise ValueError("Value with key '{}'".format(key)
                                 + " must match '{}'.".format(value)
                                 + " Got '{}' instead.".format(config[key]))
        if rules is None:
            return True
        elif any(not fn(config, key, value) for fn in rules):
            raise ValueError("Key-value check failed")

        return True

    @classmethod
    def parse(cls, x: dict) -> 'Config':
        if isinstance(x, cls):
            return x
        config = cls()
        for key, value in x.items():
            if isinstance(value, dict):
                config[key] = cls.parse(value)
            else:
                config[key] = value
        return config

    @classmethod
    def merge(cls, x: dict, y: dict) -> 'Config':
        d = cls()
        for k in x.keys() | y.keys():
            if k in x and k in y:
                if (isinstance(x[k], dict)
                        and isinstance(y[k], dict)):
                    d[k] = cls.merge(x[k], y[k])
                elif isinstance(y[k], dict):
                    d[k] = y[k]
                elif isinstance(x[k], dict):
                    d[k] = y[k]
                else:
                    d[k] = y[k]
            elif k in x:
                d[k] = x[k]
            elif k in y:
                d[k] = y[k]
        return d

    def flatten(self) -> dict:
        result = {}
        config = super(Config, self)
        for key, value in config.items():

            if not isinstance(value, (Config, dict)):
                result[key] = value
                continue

            value = value.flatten()
            result.update({(key + '/' + k): v
                           for k, v in value.items()})
        return result

    def update(self, other=None, **kwargs):
        """ Update config with values from other. """
        other = Config.parse({} if other is None
                             else other).flatten()
        for key in other:
            self[key] = other[key]
        for key, value in kwargs.items():
            self[key] = value

    def pop(self, key: str, default: 'Any' = None) -> 'Any':
        if not isinstance(key, str):
            return super(Config, self).__getitem__(key)
        *other_keys, last_key = key.strip('/').split('/')
        config = self
        for ikey in other_keys:
            try:
                config = super(Config, config).__getitem__(ikey)
            except KeyError:
                return default
        return super(Config, config).pop(last_key, default)

    def get(self, key: str, default: 'Any' = None) -> 'Any':
        if not isinstance(key, str):
            return super(Config, self).__getitem__(key)
        *other_keys, last_key = key.strip('/').split('/')
        config = self
        for ikey in other_keys:
            try:
                config = super(Config, config).__getitem__(ikey)
            except KeyError:
                return default
        return super(Config, config).get(last_key, default)

    def __getitem__(self, key: str) -> 'Any':
        if not isinstance(key, str):
            return super(Config, self).__getitem__(key)
        *other_keys, last_key = key.strip('/').split('/')
        config = self
        for ikey in other_keys:
            config = super(Config, config).__getitem__(ikey)
        return super(Config, config).__getitem__(last_key)

    def __delitem__(self, key: str) -> 'Any':
        if not isinstance(key, str):
            return super(Config, self).__delitem__(key)
        *other_keys, last_key = key.strip('/').split('/')
        config = self
        for ikey in other_keys:
            config = super(Config, config).__getitem__(ikey)
        return super(Config, config).__delitem__(last_key)

    def __setitem__(self, key: str, value: 'Any'):
        if isinstance(value, dict):
            value = self.parse(value)
        if not isinstance(key, str):
            return super(Config, self).__setitem__(key, value)
        *other_keys, last_key = key.strip('/').split('/')
        config = self
        for ikey in other_keys:
            config = super(Config, config)
            if not config.__contains__(ikey):
                config.__setitem__(ikey, Config())
            ivalue = config.__getitem__(ikey)
            if not isinstance(ivalue, dict):
                config.__setitem__(ikey, Config())
            config = config.__getitem__(ikey)

        config = super(Config, config)
        if not config.__contains__(last_key):
            config.__setitem__(last_key, value)
        last_value = config.__getitem__(last_key)
        if isinstance(value, dict) and isinstance(last_value, dict):
            value = self.merge(last_value, value)
        return config.__setitem__(last_key, value)

    def __contains__(self, key: str) -> bool:
        if not isinstance(key, str):
            return super(Config, self).__contains__(key)
        *other_keys, last_key = key.strip('/').split('/')
        config = self
        for ikey in other_keys:
            try:
                config = super(Config, config).__getitem__(ikey)
            except KeyError:
                return False
        return super(Config, config).__contains__(last_key)

    def __matmul__(self, other):
        if not isinstance(other, dict):
            raise TypeError("Operands must be"
                            + " instances of dictionary")

        other = self.parse(other)
        return self.merge(self, other)

    def __rmatmul__(self, other):
        if not isinstance(other, dict):
            raise TypeError("Operands must be"
                            + " instances of dictionary")
        other = self.parse(other)
        return self.merge(other, self)

    def __add__(self, other):
        if not isinstance(other, dict):
            raise TypeError("Operands must be"
                            + " instances of dictionary")
        other = self.parse(other)
        return self.merge(self, other)

    def __radd__(self, other):
        if not isinstance(other, dict):
            raise TypeError("Operands must be"
                            + " instances of dictionary")
        other = self.parse(other)
        return self.merge(other, self)

    def __rshift__(self, other):
        if not isinstance(other, dict):
            raise TypeError("Operands must be"
                            + " instances of dictionary")
        other = self.parse(other).flatten()
        output = Config()
        for key, value in self.flatten().items():
            if key in other:
                mapped_key = str(other[key])
            else:
                mapped_key = key
            output[mapped_key] = value
        return output

    def __lshift__(self, other):
        return self.parse(other).__rshift__(self)

    def __rrshift__(self, other):
        return self.parse(other).__rshift__(self)

    def __rlshift__(self, other):
        return self.parse(other).__lshift__(self)

    def items(self, flatten=False):
        """ Return config items. """
        if flatten:
            return self.flatten().items()
        else:
            return super(Config, self).items()

    def keys(self, flatten=False):
        """ Return config keys. """
        if flatten:
            return self.flatten().keys()
        else:
            return super(Config, self).keys()

    def values(self, flatten=False):
        """ Return config values. """
        if flatten:
            return self.flatten().values()
        else:
            return super(Config, self).values()
