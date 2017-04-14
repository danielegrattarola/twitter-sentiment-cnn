"""
Contains general purpose functions that would clutter up the main source file. 
"""


def print_green(string):
    print "".join(["\033[92m", string, "\033[0m"])


def print_red(string):
    print "".join(["\033[91m", string, "\033[0m"])
