import os.path
import configparser
import numpy

DEFAULT_PATH = 'config.txt'


class Config(object):
  def __init__(self, config_path=DEFAULT_PATH):
    if config_path==None:
      config_path = DEFAULT_PATH
    if not os.path.isfile(config_path):
      print("Cannot open configuration file \""+config_path+"\"")
      exit(1)
    print("Configuration file:", config_path)

    self.configuration = configparser.RawConfigParser()
    self.configuration.read(config_path)
    self.options = dict(self.configuration.items('options'))
    self.paths   = dict(self.configuration.items('paths'))

  def get_option(self, key, default, ttype):
    if key in self.options:
      return ttype(self.options[key])
    else:
      return default

  def floats(self, x):
    return (numpy.array(sorted(list(map(float, x.split(','))))))


def get_config_path(argv):
  path = None
  paths = ([x for x in argv[1:] if x.startswith('--config=')])
  if paths:
    path = paths[-1][len('--config='):]
  return path

