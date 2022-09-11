import torch.utils.data


class Server(object):
    def __init__(self, conf, eval_dataset):
        self.conf = conf
        self.global_model = models.get_model(self.conf("model_name"))
        self.eval_loader = torch.utils.data.DataLoader(eval_dataset,
                                                       batch_size=self.conf("batch_size"),
                                                       shuffle=True)
    def model_aggregate(self):
        for name, data in self.global_model.state_dic().items():
            update_per_layer = weight_accumulator(name) * self.conf["lamda"]

