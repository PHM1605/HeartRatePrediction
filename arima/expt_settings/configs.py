import os
import data_formatters.heart_rate

class ExperimentConfig():
    def __init__(self, experiment):
        self.experiment = experiment
        self.data_folder = os.path.join('/uio/kant/ifi-ansatt-u04/hoangmph/hoangminh/data/mecs', self.experiment)
        
    @property
    def data_csv_path(self):
        csv_map = { self.experiment : 'uwb_hr_' + self.experiment + '.csv'}
        return os.path.join(self.data_folder, csv_map[self.experiment])
    
    def make_data_formatter(self):
        data_formatter_class = {'heart_rate': data_formatters.heart_rate.HeartRateFormatter}
        return data_formatter_class['heart_rate']()