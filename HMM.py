import pandas as pd
import argparse
import numpy as np
from numpy.random import choice

# Alexander Bareli


class Observation:
    def __init__(self, stateseq, outputseq):
        self.stateseq = stateseq   # sequence of states
        self.outputseq = outputseq  # sequence of outputs

    def __str__(self):
        return ' '.join(self.stateseq)+'\n'+' '.join(self.outputseq)+'\n'

    def __repr__(self):
        return self.__str__()

    def __len__(self):
        return len(self.outputseq)


# hmm model
class HMM:
    def __init__(self, transitions={}, emissions={}):
        """creates a model from transition and emission probabilities"""
        # Both of these are dictionaries of dictionaries. e.g. :
        # {'#': {'C': 0.814506898514, 'V': 0.185493101486},
        #  'C': {'C': 0.625840873591, 'V': 0.374159126409},
        #  'V': {'C': 0.603126993184, 'V': 0.396873006816}}

        self.transitions = transitions
        self.emissions = emissions

    def load(self, basename):
        trans = open(basename + ".trans", "r")
        transitions = {}
        for line in trans:
            data = line.strip().split(" ")
            if transitions.get(data[0]) is None:
                transitions.update({data[0]: {data[1]: float(data[2])}})
            else:
                temp = transitions.get(data[0])
                temp.update({data[1]: float(data[2])})
                transitions.update({data[0]: temp})

        emiss = open(basename + ".emit", "r")
        emissions = {}
        for line in emiss:
            data = line.strip().split(" ")
            if emissions.get(data[0]) is None:
                emissions.update({data[0]: {data[1]: float(data[2])}})
            else:
                temp = emissions.get(data[0])
                temp.update({data[1]: float(data[2])})
                emissions.update({data[0]: temp})

        self.transitions = transitions
        self.emissions = emissions

    def generate(self, n):
        startState = self.transitions.get("#")
        state = choice(list(startState.keys()), p=list(startState.values()))
        startEmission = self.emissions.get(state)
        emission = choice(list(startEmission.keys()), p=list(startEmission.values()))
        states = [state]
        emissions = [emission]
        for i in range(1, n):
            new_state = choice(list(self.transitions.get(state).keys()), p=list(self.transitions.get(state).values()))
            new_emission = choice(list(self.emissions.get(new_state).keys()),
                                  p=list(self.emissions.get(new_state).values()))
            states.append(new_state)
            emissions.append(new_emission)
            state = new_state
        observed = Observation(states, emissions)
        return observed

    def generate_matrix(self, lines):
        matrix = np.zeros((len(list(self.transitions.keys())), len(lines.split()) + 1))
        matrix[0][0] = 1.0
        return matrix

    def forward(self, observation):
        observations = open(observation + '.obs', 'r')
        for line in observations.readlines():
            if len(line) > 1:
                letters = line.strip().split()
                matrix = self.generate_matrix(line)
                index = 1

                for key in self.emissions.keys():
                    if self.emissions.get(key).get(letters[0]) is not None:
                        matrix[index][1] = self.transitions.get('#').get(key) * self.emissions.get(key).get(letters[0])
                    index += 1
                letters.insert(0, '-')
                panda = pd.DataFrame(matrix, index=list(self.transitions.keys()))

                for i in range(2, len(letters)):
                    for key in self.emissions.keys():
                        sum = 0
                        for keyed in self.emissions.keys():
                            if self.emissions.get(key).get(letters[i]) is not None:
                                sum += (panda[i-1][keyed] * self.transitions.get(keyed).get(key) *
                                        self.emissions.get(key).get(letters[i]))
                        panda[i][key] = sum

                best_state = panda[panda.columns[-1]].idxmax()
                print(letters[1:])
                print("The best final state given observation is: ", best_state, '\n')
                # print(panda.set_axis(letters, axis=1))

    def viterbi(self, observation):
        observations = open(observation + '.obs', 'r')
        for line in observations.readlines():
            if len(line) > 1:
                letters = line.strip().split()
                matrix = self.generate_matrix(line)
                backpointer = self.generate_matrix(line)
                index = 1

                for key in self.emissions.keys():
                    if self.emissions.get(key).get(letters[0]) is not None:
                        matrix[index][1] = self.transitions.get('#').get(key) * self.emissions.get(key).get(letters[0])
                    index += 1
                letters.insert(0, '-')
                panda = pd.DataFrame(matrix, index=list(self.transitions.keys()))
                back = pd.DataFrame(backpointer, index=list(self.transitions.keys()))

                for i in range(2, len(letters)):
                    for key in self.emissions.keys():
                        sums = {}
                        for keyed in self.emissions.keys():
                            if self.emissions.get(key).get(letters[i]) is not None:
                                sum = (panda[i-1][keyed] * self.transitions.get(keyed).get(key) *
                                       self.emissions.get(key).get(letters[i]))
                                sums.update({sum: keyed})
                        if len(sums) != 0:
                            panda[i][key] = max(sums)
                            indexes = list(back.index)
                            back[i][key] = indexes.index(sums.get(max(sums)))
                        else:
                            panda[i][key] = 0
                            back[i][key] = 0

                likely_states = []
                indexes = list(back.index)
                best_state = panda[panda.columns[-1]].idxmax()
                likely_states.append(best_state)

                for i in range(len(letters)-1, 1, -1):
                    cur_state = back.get(i).get(best_state)
                    state = indexes[int(cur_state)]
                    likely_states.append(state)
                    best_state = state

                likely_states.reverse()
                print(likely_states)
                print(letters[1:], '\n')
                # print(panda.set_axis(letters, axis=1))
                # print(back.set_axis(letters, axis=1))


if __name__ == '__main__':
    parser = argparse.ArgumentParser("HMM")
    parser.add_argument("hmm")
    parser.add_argument("--generate", type=int)
    parser.add_argument("--forward")
    parser.add_argument("--viterbi")
    args = parser.parse_args()

    model = HMM()
    if args.hmm is not None:
        model.load(args.hmm)
    if args.generate is not None:
        print(model.generate(args.generate))
    if args.forward is not None:
        model.forward(args.forward)
    if args.viterbi is not None:
        model.viterbi(args.viterbi)
