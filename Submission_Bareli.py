import carnet
import alarm
import HMM

# Author: Alexander Bareli

if __name__ == "__main__":

    print("\n Alarm Bayesian Network Queries: \n")
    alarm.main()

    print("\n Carnet Bayesian Network Queries: \n")
    carnet.main()

    print("\n Hidden Markov Model: \n")
    model = HMM.HMM()
    model.load("partofspeech.browntags.trained")
    print("Randomly generated observation: \n")
    print(model.generate(20))
    print("Forward Algorithm: \n")
    model.forward('ambiguous_sents')
    print("Viterbi Algorithm: \n")
    model.viterbi('ambiguous_sents')
