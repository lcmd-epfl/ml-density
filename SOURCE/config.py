import configparser

class Config(object):
  def __init__(self):
    self.configuration = configparser.RawConfigParser()
    self.configuration.read('config.txt')
    self.options = dict(self.configuration.items('options'))
    self.paths   = dict(self.configuration.items('paths'))

  def get_option(self, key, default, ttype):
    if key in self.options:
      return ttype(self.options[key])
    else:
      return default

