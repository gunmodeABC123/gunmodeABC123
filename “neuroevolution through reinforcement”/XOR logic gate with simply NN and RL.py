import numpy as np
import random


POP_SIZE = 50
INPUT_SIZE = 2
HIDDEN_SIZE = 3
OUTPUT_SIZE = 1
GENERATIONS = 100
MUTATION_RATE = 0.1


def logic_function(x, y):
    return x ^ y  

class TinyBrain:
    def __init__(self):
        self.w1 = np.random.randn(INPUT_SIZE, HIDDEN_SIZE)
        self.b1 = np.random.randn(HIDDEN_SIZE)
        self.w2 = np.random.randn(HIDDEN_SIZE, OUTPUT_SIZE)
        self.b2 = np.random.randn(OUTPUT_SIZE)

    def forward(self, x):
        h = np.tanh(np.dot(x, self.w1) + self.b1)
        o = self.sigmoid(np.dot(h, self.w2) + self.b2)
        return o

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def clone_and_mutate(self):
        clone = TinyBrain()
        clone.w1 = self.mutate_array(self.w1)
        clone.b1 = self.mutate_array(self.b1)
        clone.w2 = self.mutate_array(self.w2)
        clone.b2 = self.mutate_array(self.b2)
        return clone

    def mutate_array(self, arr):
        new_arr = arr.copy()
        mutation_mask = np.random.rand(*arr.shape) < MUTATION_RATE
        new_arr[mutation_mask] += np.random.randn(*arr.shape)[mutation_mask]
        return new_arr

def evaluate(brain):
    score = 0
    for x in [0, 1]:
        for y in [0, 1]:
            input_vec = np.array([x, y])
            prediction = brain.forward(input_vec)[0]
            output = 1 if prediction > 0.5 else 0
            if output == logic_function(x, y):
                score += 1
    return score  

population = [TinyBrain() for _ in range(POP_SIZE)]

for gen in range(GENERATIONS):
    scores = [evaluate(brain) for brain in population]
    best_idx = np.argmax(scores)
    best_brain = population[best_idx]
    max_score = scores[best_idx]

    print(f"Generation {gen} | Best Score: {max_score}/4")

    population = [best_brain.clone_and_mutate() for _ in range(POP_SIZE)]

print("\nFinal Logic Brain:")
for x in [0, 1]:
    for y in [0, 1]:
        pred = best_brain.forward(np.array([x, y]))[0]
        print(f"Input: {x}, {y} | Predicted: {int(pred > 0.5)} | Expected: {logic_function(x, y)}")
