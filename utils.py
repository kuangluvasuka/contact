from datetime import datetime


def time_string():
  return datetime.now().strftime("%y%m%d_%H%M")
