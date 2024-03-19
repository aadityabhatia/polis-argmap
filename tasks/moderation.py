from .task import Task

class Moderation(Task):
    def __init__(self, param):
        self.param = param

    def run(self):
        print(f"Running Generation with param: {self.param}")
