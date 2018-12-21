import os


class Utils(object):
    @classmethod
    def printProgressBar(cls, iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█'):
        """
        Call in a loop to create terminal progress bar
        @params:
            iteration   - Required  : current iteration (Int)
            total       - Required  : total iterations (Int)
            prefix      - Optional  : prefix string (Str)
            suffix      - Optional  : suffix string (Str)
            decimals    - Optional  : positive number of decimals in percent complete (Int)
            length      - Optional  : character length of bar (Int)
            fill        - Optional  : bar fill character (Str)
        """
        percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
        filledLength = int(length * iteration // total)
        bar = fill * filledLength + '-' * (length - filledLength)
        print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end='\r')
        # Print New Line on Complete
        if iteration == total:
            print()

    @classmethod
    def get_label(cls, log, label_string, separator, label_pos):
        return label_string.split(separator)[label_pos]

    @classmethod
    def pre_process_raw_string(cls, raw_string):
        return raw_string.strip()

    @classmethod
    def join_paths(cls, path1, path2):
        return os.path.join(path1, path2)

    @classmethod
    def count_items_in_dir(cls, path):
        return len(os.listdir(path))

    @classmethod
    def count_item_in_file(cls, path):
        return sum(1 for line in open(path))
