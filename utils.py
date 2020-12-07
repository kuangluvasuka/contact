from datetime import datetime


def time_string():
  return datetime.now().strftime("%b.%d_%H.%M")
