# -----------------------------------------------------------
# Single Quantum Dot Simulator that is used to Generate Training Examples for a CNN.
#
# (C) 2020 Joel Pendleton, London, UK
# Released under MIT license
# email joel.pendleton@quantummotion.tech
# -----------------------------------------------------------
from multiprocessing import Pool
from multiprocessing import freeze_support
from helper import Helper
import sys



if __name__ == "__main__":
    helper = Helper()
    try:

        generate_or_augment = sys.argv[1]
        if generate_or_augment == '--help':
            print("Make sure when you call this file you pass a flag.\n"
                  "--simulate generates training examples\n"
                  "--augment generates more training examples by augmenting existing examples\n"
                  "--both does the equivalent of --simulate followed by --augment\n"
                  "\nFor the --simulate and --both flags you need to also pass another argument, "
                  "the number of training examples"
                  "\nyou wish to generate. For -a no other arguments are required.")

        # -t = generate training examples, -a = augment training examples, -b = both generate & augment training examples

        elif generate_or_augment == "--simulate" or generate_or_augment == "--both":
            try:
                try:
                    number_of_examples = int(sys.argv[2])
                except IndexError:
                    print("An index error has occured. Make sure when you call this file you pass a flag.\n"
                          "--simulate generates training examples\n"
                          "--augment generates more training examples by augmenting existing examples\n"
                          "--both does the equivalent of --simulate followed by --augment\n"
                          "\nFor the --simulate and --both flags you need to also pass another argument, "
                          "the number of training examples"
                          "\nyou wish to generate. For --augment no other arguments are required.")
                print("Generating training data...")
                helper.generate_examples(number_of_examples)

                if generate_or_augment == "--both":
                    print("Augmenting training data...")
                    helper.augment_examples()



            except TypeError:

                print("A TypeError error has occurred.\n"
                      "Make sure when you call the program with the flag --simulate or --both you also pass a number"
                      "\n - the number of training examples you want to generate.\n"
                      "E.g. python main.py --simulate 1000\n"
                      "This generates 1000 training examples.")

        elif generate_or_augment == "--augment":
            print("Augmenting training data...")
            helper.augment_examples()

        else:
            print("Make sure when you call this file you pass one of the following flags as your first argument.\n"
                  "--simulate generates training examples\n"
                  "--augment generates more training examples by augmenting existing examples\n"
                  "--both does the equivalent of --simulate followed by --augment\n"
                  "\nFor the --simulate and --both flags you need to also pass another argument,"
                  " the number of training examples"
                  "\nyou wish to generate. For -a no other arguments are required.")


    except IndexError:

        print("An index error has occured. Make sure when you call this file you pass a flag.\n"
              "--simulate generates training examples\n"
              "--augment generates more training examples by augmenting existing examples\n"
              "--both does the equivalent of --simulate followed by --augment\n"
              "\nFor the --simulate and --both flags you need to also pass another argument, the number of training examples"
              "\nyou wish to generate. For --augment no other arguments are required.")


