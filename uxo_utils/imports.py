import os
file_loc = os.path.abspath(__file__).split(os.path.sep)
data_dir = os.path.sep.join(file_loc[:-2]+ ["data-blacktusk"])
