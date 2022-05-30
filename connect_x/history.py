from matplotlib import pyplot as plt

class History():
    def __init__(self):
        self.history = {}
    
    def add(self, key, value):
        if (key not in self.history):
            self.history[key] = []
        self.history[key].append(value)
    
    def add_tuples(self, tuples):
        for t in tuples:
            self.add(t[0], t[1])
    
    def get(self, key):
        return self.history[key]
    
    def plot(self, key):
        plt.plot(self.history[key])
        plt.show()