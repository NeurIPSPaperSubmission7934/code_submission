# Copyright (c) 2018 Copyright holder of the paper Generative Adversarial Model Learning
# submitted to NeurIPS 2019 for review
# All rights reserved.

class TorchModel(object):

    @property
    def normalized_input(self):
        raise NotImplementedError

    @property
    def normalized_output(self):
        raise NotImplementedError
