# -----------------------------------------------------------
# Single Quantum Dot Simulator that is used to Generate Training Examples for a CNN.
#
# (C) 2020 Joel Pendleton, London, UK
# Released under MIT license
# email joel.pendleton@quantummotion.tech
# -----------------------------------------------------------

from helper import Helper
import sys

# Flag 1 -> -Simulate or -Help
# Flag 2 -> Number of training examples
# Flag 3 -> Number of validation examples
# Flag 4 -> Noise or don't pass for no noise

# python main.py -Simulate 1000 1000 -noise

if __name__ == "__main__":
    helper = Helper()

    try:
        argument_1 = sys.argv[1]
        if argument_1 == '-help':
            print("Usage: python main.py [-help | -simulate] [<training-size>] [<validation-size>] [--noise]")
            print("\nThe first argument should be either -help or -simulate."
                  "\nIf you use -simulate be sure to pass values for training-size and validation-size."
                  "\ntraining-size and validation-size are the the number of examples in your training and validation"
                  " set, respectively. They must be integers"
                  "\nIf you want noise on your examples add the --noise flag at the end of the line.")
        elif argument_1 == "-simulate":
            try:
                argument_2 = int(sys.argv[2]) # training-size
                argument_3 = int(sys.argv[3]) # validation-size
                try:
                    argument_4 = sys.argv[4] # --noise
                    if argument_4 == "--noise":
                        noise = True
                except IndexError:
                    noise = False
                    pass

                if noise:
                    print("Generating training data with added noise...")
                else:
                    print("Generating training data...")
                helper.noise = noise
                if argument_2 > 0:
                    helper.path = "./data/train/"
                    helper.generate_examples(argument_2) # generate training set
                    print("Training set saved to", helper.path)
                if argument_3 > 0:
                    helper.path = "./data/val/"
                    helper.generate_examples(argument_3)  # generate validation set
                    print("Validation set saved to", helper.path)

            except (IndexError, TypeError):
                print("Make sure to pass the required arguments."
                      "\nTo see how to do this call python main.py -help")
        else:
            print("Make sure to pass the required arguments."
                  "\nTo see how to do this call python main.py -help")

    except IndexError:
        print("Make sure to pass the required arguments."
              "\nTo see how to do this call python main.py -help")





