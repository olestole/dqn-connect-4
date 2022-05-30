from matplotlib import pyplot as plt

class History():
    def __init__(self):
        self.history = {}
    
    def add(self, key, value):
        if (key not in self.history):
            self.history[key] = []
        self.history[key].append(value)
    
    def get(self, key):
        return self.history[key]
    
    def plot(self, key):
        plt.plot(self.history[key])
        plt.show()