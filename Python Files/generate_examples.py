import os
import sys

try:

    generate_or_augment = sys.argv[1]
    if generate_or_augment == 'help':
        print("Make sure when you call this file you pass a flag.\n"
              "-t generates training examples\n"
              "-a generates more training examples by augmenting existing examples\n"
              "-b does the equivalent of -t followed by -a\n"
              "\nFor the -t and -b flags you need to also pass another argument, the number of training examples"
              "\nyou wish to generate. For -a no other arguments are required.")

    # -t = generate training examples, -a = augment training examples, -b = both generate & augment training examples

    elif generate_or_augment == "-t" or generate_or_augment == "-b":
        try:
            number_of_examples = sys.argv[2]
            os.system('python simulation.py {0}'.format(number_of_examples))

        except:

            print("An error has occurred.\n"
                  "Make sure when you call the program with the flag -t or -b you also pass a number"
                  "\n - the number of training examples you want to generate.\n"
                  "E.g. python generate_examples.py -t 1000\n"
                  "This generates 1000 training examples.")

        if generate_or_augment == "-b":
            os.system('python augmentation.py')
    elif generate_or_augment == "-a":
        os.system('python augmentation.py')

except:

    print("An error occured. Make sure when you call this file you pass a flag.\n"
          "-t generates training examples\n"
          "-a generates more training examples by augmenting existing examples\n"
          "-b does the equivalent of -t followed by -a\n"
          "\nFor the -t and -b flags you need to also pass another argument, the number of training examples"
          "\nyou wish to generate. For -a no other arguments are required.")
