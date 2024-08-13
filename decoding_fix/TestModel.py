from decoding_fix.BaseModel import BaseModel


class TestModel(BaseModel):

    def __init__(self):
        super().__init__()
        self.model_name = "Test"

    def load_model(self, model_path):
        pass
