
class DataConfig:
    data_name = ""
    root_dir = ""
    label_transform = "norm"
    def get_data_config(self, data_name):
        self.data_name = data_name
        if data_name == 'LEVIR':
            self.root_dir = 'path to LEVIR-CD256'
        elif data_name == 'DSIFN':
            self.root_dir = 'path to DSIFN-CD256'
        elif data_name == 'CDD':
            self.root_dir = 'path to CDD-256'
        elif data_name == 'CMU':
            self.root_dir = 'path to CMU-CD256'
        else:
            raise TypeError('%s has not defined' % data_name)
        return self


if __name__ == '__main__':
    data = DataConfig().get_data_config(data_name='LEVIR')
    print(data.data_name)
    print(data.root_dir)
    print(data.label_transform)

