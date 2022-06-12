from pprint import pprint
import string
import numpy as np
import math
import functools
import itertools
import random
import ipaddress
import socket
import unittest

from regex import F


class Kata:
    def __init__(self, inp):
        self.inp = inp

    def rotate_13_func(self, letter):
        lower = string.ascii_lowercase
        upper = string.ascii_uppercase
        ind = lower.index(letter) if letter in lower else upper.index(letter)
        for i in range(13):
            ind += 1
            if ind == 26:
                ind = 0
        return lower[ind] if letter in lower else upper[ind]

    def rot_13(self):
        lower = string.ascii_lowercase
        upper = string.ascii_uppercase
        return "".join(
            [
                self.rotate_13_func(x) if x in lower or x in upper else x
                for x in list(self.inp)
            ]
        )

    def interpret(self, function):

        if function == "rot13":
            return self.rot_13()


def solution(string, markers):

    # print('solution = [{}] and markers = {}'.format(string, markers))
    new_arr = []
    ind = 0
    curr_string = ""
    reached_comment = False
    while ind < len(string):
        curr_letter = string[ind]
        if curr_letter == "\n":
            ## append curr string, reset curr string
            new_arr.append(curr_string.strip() + "\n")
            curr_string = ""
            reached_comment = False
        elif curr_letter in markers:
            reached_comment = True
        elif not reached_comment:
            curr_string += curr_letter
        ind += 1
    if curr_string != "":
        new_arr.append(curr_string.strip())
    return "".join(new_arr)


def count_letters(word):
    letters = {}
    condensed_letters = [x for x in set(list(word))]
    for eachletter in condensed_letters:
        letters[eachletter] = word.count(eachletter)
    return letters


def compare_words(baseword, word2):
    basewordlettercount = count_letters(baseword)
    word2lettercount = count_letters(word2)
    for eachletter in basewordlettercount:
        if eachletter not in word2lettercount:
            return False
        elif basewordlettercount[eachletter] > word2lettercount[eachletter]:
            return False
        elif word2lettercount[eachletter] > basewordlettercount[eachletter]:
            return False
    for eachletter in word2lettercount:
        if eachletter in word2lettercount and eachletter not in basewordlettercount:
            return False
    return True


def anagrams(word, words):
    return [x for x in words if compare_words(word, x)]


def domain_name(url: str):
    url = url.replace("http://", "").replace("www.", "").replace("https://", "")
    return url.split(".")[0]


def zero(func=None):
    if func:
        return func(0)
    return lambda: 0


def one(func=None):
    if func:
        return func(1)
    return lambda: 1


def two(func=None):
    if func:
        return func(2)
    return lambda: 2


def three(func=None):
    if func:
        return func(3)
    return lambda: 3


def four(func=None):
    if func:
        return func(4)
    return lambda: 4


def five(func=None):
    if func:
        return func(5)
    return lambda: 5


def six(func=None):
    if func:
        return func(6)
    return lambda: 6


def seven(func=None):
    if func:
        return func(7)
    return lambda: 7


def eight(func=None):
    if func:
        return func(8)
    return lambda: 8


def nine(func=None):
    if func:
        return func(9)
    return lambda: 9


def plus(func=None):
    return lambda x: x + func()


def minus(func=None):
    return lambda x: x - func()


def times(func=None):
    return lambda x: x * func()


def divided_by(func=None):
    return lambda x: x // func()


def last_digit(n1, n2):
    count = 0
    number = "{}".format(n1)
    pattern = []
    orig_numb = n1
    while True:
        number = int(number) * orig_numb
        number = str(number)[-1]
        if number not in pattern:
            pattern.append(number)
        else:
            break
    return number[-1]


def int32_to_ip(number):
    the_bin = bin(number)[2:].zfill(32)
    first, second, third, fourth = (
        the_bin[0:8],
        the_bin[8:16],
        the_bin[16:24],
        the_bin[24:],
    )
    return "{}.{}.{}.{}".format(
        int(first, 2), int(second, 2), int(third, 2), int(fourth, 2)
    )


def arrange(strng):
    words = strng.split(" ")
    checker = False
    for i in range(len(words) - 1):
        word_one = words[i]
        word_two = words[i + 1]
        if not checker and len(word_one) > len(word_two):
            words[i], words[i + 1] = words[i + 1], words[i]
        elif checker and len(word_one) < len(word_two):
            words[i], words[i + 1] = words[i + 1], words[i]
        checker = not checker
    for i in range(len(words)):
        if i % 2 != 0:
            words[i] = words[i].upper()
        else:
            words[i] = words[i].lower()
    return " ".join(words)


def freq_count(arr, element):
    all_elements = nested_iterator(arr)
    nested_map = {}
    max_level = 0
    for eachelem in all_elements:
        level = eachelem.split(" ")[1]
        max_level = max(max_level, int(level))
        elem = int(eachelem.split(" ")[0])
        if elem == element:
            if level in nested_map:
                nested_map[level] = nested_map[level] + 1
            else:
                nested_map[level] = 1
    levels = []
    for i in range(max_level + 1):
        if i not in nested_map:
            levels.append([i, 0])
        else:
            levels.append([str(i), nested_map[i]])
    return levels


def nested_iterator(arr, nestedLevel=0):

    elements = []
    for eachelem in arr:
        if type(eachelem) is list:
            elements += nested_iterator(eachelem, nestedLevel + 1)
        else:
            elements.append("{} {}".format(eachelem, nestedLevel))
    return elements


def is_prime(anum):
    if anum in [1, 2, 3, 5, 7]:
        return True
    else:
        if anum % 2 == 0 or anum % 3 == 0 or anum % 5 == 0:
            return False
        else:
            for i in range(2, round(math.ceil(math.sqrt(anum))) + 1):
                if anum % i == 0:
                    return False
            return True


def prime_factors(anum):

    start = 2
    factors = []
    while anum > 1:
        if anum % start == 0:
            while anum % start == 0:
                anum /= start
                factors.append(start)
            start = 2
            continue
        elif is_prime(anum):
            factors.append(round(anum))
            return factors
        start += 1
    return factors


def is_economical(number):
    factors = prime_factors(number)
    total = 0
    factors_set = set(factors)
    numbers = []
    for eachnumber in factors_set:
        if factors.count(eachnumber) > 1:
            numbers.append("{}{}".format(eachnumber, factors.count(eachnumber)))
        else:
            numbers.append("{}".format(eachnumber))
    digit_length = len("".join(numbers))
    return (
        "Equidigital"
        if digit_length == len(str(number))
        else "Frugal"
        if digit_length < len(str(number))
        else "Wasteful"
    )


def is_divisible(n, x, y):

    return n % x == 0 and n % y == 0


def is_pangram(s):
    s = s.lower()
    lower_ = string.ascii_lowercase
    return "".join([x for x in set([x for x in s if x in lower_])]) == lower_


def dirReduc(arr):
    index = 0
    while index < len(arr):
        the_direction = arr[index]
        if the_direction in "NORTHSOUTH":
            if the_direction == "NORTH":
                if index < len(arr) - 1 and arr[index + 1] == "SOUTH":
                    del arr[index + 1]
                    del arr[index]
                    index = -1
            elif the_direction == "SOUTH":
                if index < len(arr) - 1 and arr[index + 1] == "NORTH":
                    del arr[index + 1]
                    del arr[index]
                    index = -1
        elif the_direction in "EASTWEST":
            if the_direction == "EAST":
                if index < len(arr) - 1 and arr[index + 1] == "WEST":
                    del arr[index + 1]
                    del arr[index]
                    index = -1

            elif the_direction == "WEST":
                if index < len(arr) - 1 and arr[index + 1] == "EAST":
                    del arr[index + 1]
                    del arr[index]
                    index = -1
        index += 1
    return arr


def unique_in_order(seq):

    data = []
    for k, g in itertools.groupby(seq):
        data.append(k)
    return data


def sum_str(a, b):

    a = a if a != "" else "0"
    b = b if b != "" else "0"
    return int(a) + int(b)


def snail(snail_map):
    if snail_map == []:
        return []
    else:
        directions = ["r", "d", "l", "u"]
        x = 0
        y = 0
        one_count = 0
        arr_count = len(snail_map) * len(snail_map)
        dir_ind = 0
        values = []
        while one_count < arr_count:
            if directions[dir_ind] == "r":
                # move right until reach -1
                values.append(snail_map[x][y])
                snail_map[x][y] = -1
                y += 1
                one_count += 1
                if y == len(snail_map[0]) or snail_map[x][y] == -1:
                    y -= 1
                    x += 1
                    dir_ind += 1
            elif directions[dir_ind] == "d":
                values.append(snail_map[x][y])
                snail_map[x][y] = -1
                x += 1
                one_count += 1
                if x == len(snail_map) or snail_map[x][y] == -1:
                    x -= 1
                    y -= 1
                    dir_ind += 1
            elif directions[dir_ind] == "l":
                values.append(snail_map[x][y])
                snail_map[x][y] = -1
                y -= 1
                one_count += 1
                if y == -1 or snail_map[x][y] == -1:
                    y += 1
                    x -= 1
                    dir_ind += 1
            elif directions[dir_ind] == "u":
                values.append(snail_map[x][y])
                snail_map[x][y] = -1
                x -= 1
                one_count += 1
                if x == -1 or snail_map[x][y] == -1:
                    x += 1
                    y += 1
                    dir_ind = 0
        return values


def find_needle(haystack):

    return "found the needle at position {}".format(haystack.index("needle"))


def find(n):
    return sum([x if x % 3 == 0 and x % 5 == 0 else 0 for x in range(3, n + 1)])


def yoda_talk(x):
    words = x.split(" ")
    verb_ind = -1
    obj_ind = 0
    sub_ind = -1
    for i in range(len(words)):
        the_word = words[i].lower()
        if verb_ind == -1 and the_word in verbs:
            verb_ind = i
        elif sub_ind == -1 and the_word in subjects:
            sub_ind = 0
        elif verb_ind != -1 and sub_ind != -1:
            obj_ind = i
            break
    obj_part = words[obj_ind:]
    subject_part = words[sub_ind:verb_ind]
    verb_part = words[verb_ind:obj_ind]
    return "{}, {} {}".format(
        " ".join(obj_part).capitalize(),
        " ".join(subject_part).lower(),
        " ".join(verb_part).lower(),
    )


def get_length_of_missing_array(array_of_arrays):

    lens = sorted([len(x) if x != None else 0 for x in array_of_arrays])
    if len(lens) == 0 or 0 in lens:
        return 0
    for i in range(len(lens)):
        eachnumber = lens[i]
        if i == (len(lens) - 1):
            return 0
        elif eachnumber + 1 not in lens:
            return eachnumber + 1


def deep_count(a):

    count = 0
    for eachelem in a:
        if type(eachelem) is not list:
            count += 1
        else:
            count += 1 + deep_count(eachelem)
    return count


def balance(left, right):
    return sum([3 if x == "?" else 2 for x in left]) == sum(
        [3 if x == "?" else 2 for x in right]
    )


def ip_to_num(ip: str) -> int:
    """
    Takes a string representing an ip address, and returns an integer of all the ip addresses parts converted to binary octets, joined, and converted to base 10

    Args:
        ip (str): ip address

    Returns:
        int: ip address's parts converted to binary octets and joined together and converted to base 10
    """
    parts = ip.split(".")
    bins = []
    for eachpart in parts:
        bins.append(bin(int(eachpart))[2:].rjust(8, "0"))
    joined_bin = "".join(bins)
    return int(joined_bin, 2)


def num_to_ip(num: int) -> str:
    """
    Takes an integer, and converts it to it's ip address equivalent

    Args:
        num (int): integer number used to return ip address equivalent

    Returns:
        str: ip address
    """
    padded_bin = bin(num)[2:].rjust(32, "0")
    part1, part2, part3, part4 = (
        padded_bin[0:8],
        padded_bin[8:16],
        padded_bin[16:24],
        padded_bin[24:],
    )
    return "{}.{}.{}.{}".format(
        int(part1, 2), int(part2, 2), int(part3, 2), int(part4, 2)
    )


def simple_multiplication(number):
    return number * 8 if number % 2 == 0 else number * 9


class Sudoku(object):
    def __init__(self, data):
        widths = [x for x in set([len(x) for x in data])]
        self.is_valid_board = False
        if len(widths) != 1:
            self.is_valid_board = False
        elif round(math.sqrt(widths[0])) != math.sqrt(widths[0]):
            self.is_valid_board = False
        else:
            self.is_valid_board = True
        length = math.sqrt(len(data))
        if round(length) != length:
            self.is_valid_board = False
        else:
            self.is_valid_board = True
        self.width = widths[0]
        self.length = len(data)
        self.board = data
        self.sub_box_length = round(length)

    def is_valid(self):
        if not self.is_valid_board:
            return False
        else:
            return (
                self.check_rows() and self.check_columns() and self.check_sub_squares()
            )

    def is_valid_sequence(self, data):
        return (
            len([x for x in set(data)]) == self.width
            and min(data) == 1
            and max(data) == self.length
            and len([x for x in data if type(x) is not int]) == 0
        )

    def check_rows(self):
        for eachrow in self.board:
            if self.is_valid_sequence(eachrow):
                continue
            else:
                return False
        return True

    def check_columns(self):
        column = []
        for i in range(self.length):
            for j in range(self.width):
                column.append(self.board[j][i])
            if not self.is_valid_sequence(column):
                return False
            column = []
        return True

    def check_sub_squares(self):
        sub_box = []
        for es in range(0, self.length, self.sub_box_length):
            ## goes down rows
            for i in range(0, self.width, self.sub_box_length):
                ## goes right
                for j in range(es, es + self.sub_box_length):
                    ## goes down sub box
                    for k in range(i, i + self.sub_box_length):
                        sub_box.append(self.board[j][k])
                if not self.is_valid_sequence(sub_box):
                    return False
                sub_box = []
        return True


def traverse_TCP_states(_events):

    curr_state = "CLOSED"
    events = {
        "CLOSED": {
            "APP_PASSIVE_OPEN": "LISTEN",
            "APP_ACTIVE_OPEN": "SYN_SENT",
        },
        "LISTEN": {
            "RCV_SYN": "SYN_RCVD",
            "APP_SEND": "SYN_SENT",
            "APP_CLOSE": "CLOSED",
        },
        "SYN_RCVD": {
            "APP_CLOSE": "FIN_WAIT_1",
            "RCV_ACK": "ESTABLISHED",
        },
        "SYN_SENT": {
            "RCV_SYN": "SYN_RCVD",
            "RCV_SYN_ACK": "ESTABLISHED",
            "APP_CLOSE": "CLOSED",
        },
        "ESTABLISHED": {"APP_CLOSE": "FIN_WAIT_1", "RCV_FIN": "CLOSE_WAIT"},
        "FIN_WAIT_1": {
            "RCV_FIN": "CLOSING",
            "RCV_FIN_ACK": "TIME_WAIT",
            "RCV_ACK": "FIN_WAIT_2",
        },
        "CLOSING": {"RCV_ACK": "TIME_WAIT"},
        "FIN_WAIT_2": {"RCV_FIN": "TIME_WAIT"},
        "TIME_WAIT": {"APP_TIMEOUT": "CLOSED"},
        "CLOSE_WAIT": {"APP_CLOSE": "LAST_ACK"},
        "LAST_ACK": {"RCV_ACK": "CLOSED"},
    }

    for eachevent in _events:
        possible_events = events[curr_state]
        if eachevent not in possible_events:
            return "ERROR"
        else:
            curr_state = possible_events[eachevent]
    return curr_state


def next_smaller(n):
    str_number = str(n)
    l_digit = len(str_number) - 2
    x = 0
    li_str = list(str_number)
    while l_digit >= 0:
        right_digit = li_str[l_digit + 1]
        if int(right_digit) < int(li_str[l_digit]):
            ## found x
            x = l_digit
            break
        l_digit -= 1
    r_digit = x + 1
    max_digit = -1
    y = 0
    while r_digit < len(str_number):
        if int(li_str[r_digit]) < int(li_str[x]) and int(li_str[r_digit]) > max_digit:
            y = r_digit
            max_digit = int(li_str[r_digit])
        r_digit += 1
    ## found x and y
    li_str[x], li_str[y] = li_str[y], li_str[x]
    y, x = x, y
    y_slice = [str(x) for x in sorted([int(x) for x in li_str[y + 1 :]], reverse=True)]
    li_str = li_str[: y + 1] + y_slice
    result = int("".join(li_str))
    return -1 if result >= n or len(str(result)) != len(str(n)) else result


def name(s: list):
    alpha = " abcdefghijklmnopqrstuvwxyz"
    sum_indexes = [sum([alpha.index(y) for y in x]) for x in s]
    print(sum_indexes)


def get_planet_name(id):
    names = {
        1: "Mercury",
        2: "Venus",
        3: "Earth",
        4: "Mars",
        5: "Jupiter",
        6: "Saturn",
        7: "Uranus",
        8: "Neptune",
    }
    if id not in names:
        return ""
    else:
        return names[id]


# TODO: https://www.codewars.com/kata/5868a68ba44cfc763e00008d/train/python


def convert_bits_2d(bits_2d):
    bits_2d_string_rows = []
    for eachrow in bits_2d:
        bits_2d_string_rows.append("".join([str(x) for x in eachrow]))
    result = "\r\n".join(bits_2d_string_rows)
    return result


def print_stats(stats, toggle):
    if toggle:
        print(
            "code = {} and iterations = {} and width = {} and height = {}".format(
                stats[0], stats[1], stats[2], stats[3]
            )
        )


def convert_bits_2d(bits_2d):
    bits_2d_string_rows = []
    for eachrow in bits_2d:
        bits_2d_string_rows.append("".join([str(x) for x in eachrow]))
    result = "\r\n".join(bits_2d_string_rows)
    return result


def print_stats(stats, toggle):
    if toggle:
        print(
            "code = {} and iterations = {} and width = {} and height = {}".format(
                stats[0], stats[1], stats[2], stats[3]
            )
        )


def interpreter(code, iterations, width, height):
    bits = [0 for x in range(width)]
    bits_2d = []
    for y in range(height):
        bits_2d.append(bits[:])
    if iterations < 1:
        return convert_bits_2d(bits_2d)
    data_y = 0
    data_x = 0
    ind = 0
    _iter = 0
    curr_command = ""
    commands = "nesw*[]"
    code = "".join([x for x in code if x in commands])
    print_stats([code, iterations, width, height], True)
    while ind < len(code) and _iter < iterations:
        curr_command = code[ind]
        if curr_command == "n":
            ## move pointer up
            if data_y == 0:
                data_y = len(bits_2d) - 1
            else:
                data_y -= 1
            ind += 1
            _iter += 1
        elif curr_command == "e":
            ## move pointer right
            if data_x == len(bits_2d[0]) - 1:
                data_x = 0
            else:
                data_x += 1
            ind += 1
            _iter += 1
        elif curr_command == "s":
            ## move pointer down
            if data_y == len(bits_2d) - 1:
                data_y = 0
            else:
                data_y += 1
            ind += 1
            _iter += 1
        elif curr_command == "w":
            ## move pointer left
            if data_x == 0:
                data_x = len(bits_2d[0]) - 1
            else:
                data_x -= 1
            ind += 1
            _iter += 1
        elif curr_command == "*":
            ## flip bit at current pointer
            if bits_2d[data_y][data_x] == 0:
                bits_2d[data_y][data_x] = 1
            else:
                bits_2d[data_y][data_x] = 0
            ind += 1
            _iter += 1
        elif curr_command == "[":
            ## jump past matching ] if bit under current pointer is 0
            if bits_2d[data_y][data_x] == 0:
                stack = []
                ind += 1
                ## jump to matching ]
                while ind < len(code):
                    if code[ind] == "]":
                        if len(stack) == 0:
                            ind += 1
                            break
                        else:
                            stack.pop()
                    elif code[ind] == "[":
                        stack.append("[")
                    ind += 1
            else:
                ind += 1
            _iter += 1
        elif curr_command == "]":
            ## jump back to matching [ if bit under current pointer is 1
            if bits_2d[data_y][data_x] == 1:
                stack = []
                ind -= 1
                while ind >= 0:
                    if code[ind] == "[":
                        if len(stack) == 0:
                            ind += 1
                            break
                        else:
                            stack.pop()
                    elif code[ind] == "]":
                        stack.append("]")
                    ind -= 1
                _iter += 1
            else:
                ind += 1
    # pprint(bits_2d)
    ## convert to proper output
    return convert_bits_2d(bits_2d)


def palindrome(num):
    if type(num) is not int or num < 0:
        return "Not valid"
    sorted_num = "".join(sorted([x for x in str(num)]))
    pairs = [len(list(g)) for k, g in itertools.groupby(sorted_num)]
    # pprint(pairs)
    if len(pairs) == 1 and max(pairs) == 1:
        return False
    elif len(pairs) == 1 and pairs[0] % 2 != 0:
        return False
    elif len(pairs) == 1 and pairs[0] % 2 == 0:
        return True
    return len([x for x in pairs if x % 2 != 0]) <= 1 and num > 10


stack = []


def push_arg(arg):
    print("arg = {}".format(arg))
    global stack
    stack.append(arg)

    def extra_lambd(a):
        if a.__code__.co_argcount == 0:
            return a()
        return a

    return extra_lambd


def _add(x):
    stack.append(stack.pop() + stack.pop())
    if x.__code__.co_argcount == 0:
        return x()
    return x


def _sub(x):
    stack.append(stack.pop() - stack.pop())
    if x.__code__.co_argcount == 0:
        return x()
    return x


def _mul(x):
    global stack
    stack.append(stack.pop() * stack.pop())
    if x.__code__.co_argcount == 0:
        return x()
    return x


def _div(x):
    global stack
    stack.append(stack.pop() // stack.pop())
    if x.__code__.co_argcount == 0:
        return x()
    return x


def end_prog():
    global stack
    return 0 if len(stack) == 0 else stack[-1]


def start_prog(x):
    if x.__code__.co_argcount == 0:
        return x()
    return x


def ips_between(start, end):
    ## convert first to bin_equiv
    start_ip = ipaddress.ip_address(start)
    end_ip = ipaddress.ip_address(end)
    return abs(int(start_ip) - int(end_ip))


def fourth_problem():
    dimensions = input().split(" ")
    n, m = int(dimensions[0]), int(dimensions[1])
    n_arr = [int(x) for x in input().split(" ")]
    m_arr = [int(x) for x in input().split(" ")]
    matrix = []
    for i in range(len(m_arr)):
        sub_arr = []
        for j in range(len(n_arr)):
            sub_arr.append(n_arr[j] * m_arr[i])
        matrix.append(sub_arr)
    x = int(input())
    x_range = [x for x in range(len(matrix)) if x < n]
    y_range = [y for y in range(len(matrix)) if y < m]
    print("x_range = {}".format(x_range))
    print("y_range = {}".format(y_range))


def a_flip_bit(bit):
    return 1 if bit == 0 else 0


def automata(rules, initial, generations):

    copy_initial = initial[:]
    next_gen = copy_initial[:]
    for i in range(generations):
        ## calculate neighbors
        ## handle endpoints
        find_flips = []
        find_negates = []
        for i in range(1, len(next_gen)):
            if i < len(next_gen) - 1:
                ## grab neighbors
                curr_slice = [next_gen[i - 1], next_gen[i], next_gen[i + 1]]
                if curr_slice in rules:
                    ## found flips
                    find_negates.append(i - 1)
                    find_flips.append(i)
                    find_negates.append(i + 1)
        left_endpoint = [next_gen[-1], next_gen[0], next_gen[1]]
        left_test = [len(next_gen) - 1, 0, 1]
        if (
            left_endpoint in rules
            and len([x for x in left_test if x in find_flips]) < 1
        ):
            find_negates.append(len(next_gen) - 1)
            find_flips.append(0)
            find_negates.append(1)
        right_endpoint = [next_gen[-2], next_gen[-1], next_gen[0]]
        right_test = [len(next_gen) - 2, len(next_gen) - 1, 0]
        if (
            right_endpoint in rules
            and len([x for x in right_test if x in find_flips]) < 1
        ):
            find_negates.append(len(next_gen) - 2)
            find_flips.append(len(next_gen) - 1)
            find_negates.append(0)
        for eachnumber in find_flips:
            next_gen[eachnumber] = a_flip_bit(next_gen[eachnumber])
        for eachnumber in find_negates:
            next_gen[eachnumber] = 0
        find_flips = []
        find_negates = []
    return next_gen


NATO = {
    "A": "Alfa",
    "B": "Bravo",
    "C": "Charlie",
    "D": "Delta",
    "E": "Echo",
    "F": "Foxtrot",
    "G": "Golf",
    "H": "Hotel",
    "I": "India",
    "J": "Juliett",
    "K": "Kilo",
    "L": "Lima",
    "M": "Mike",
    "N": "November",
    "O": "Oscar",
    "P": "Papa",
    "Q": "Quebec",
    "R": "Romeo",
    "S": "Sierra",
    "T": "Tango",
    "U": "Uniform",
    "V": "Victor",
    "W": "Whiskey",
    "X": "Xray",
    "Y": "Yankee",
    "Z": "Zulu",
}


def process_letter(letter):
    return (
        letter
        if letter in string.punctuation or letter in " "
        else NATO[letter[0].upper()]
    )


def to_nato(words):
    result = [process_letter(x.upper()) for x in words]
    return " ".join(
        [
            x
            for x in " ".join([process_letter(x.upper()) for x in words]).split(" ")
            if x != ""
        ]
    )


def nth_fib(n):
    seq = [0, 1]
    while len(seq) < n:
        seq_len = len(seq)
        seq.append(seq[seq_len - 1] + seq[seq_len - 2])
    return seq[n - 1]


def get_domain(astr):
    return socket.getfqdn(astr)


def snakefill(number):
    num = number * number
    start_length = 1
    ct = 0
    while (start_length + start_length) <= num:
        start_length += start_length
        ct += 1
    return ct


class CoffeeShop:
    def __init__(self, name: str, menu: list, orders: list):
        self.name = name
        self.menu = menu
        self.orders = orders

    def add_order(self, order):
        if sum([1 if x["item"] == order else 0 for x in self.menu]) > 0:
            self.orders.append(order)
            return "Order added!"
        else:
            return "This item is currently unavailable!"

    def fulfill_order(self):
        if len(self.orders) > 0:
            _item = self.orders.pop(0)
            return "The {} is ready!".format(_item)
        else:
            return "All orders have been fulfilled!"

    def list_orders(self):
        return [x for x in self.orders]

    def due_amount(self):
        return round(sum(
            [
                y["price"] * self.orders.count(y["item"])
                for y in [x for x in self.menu if x["item"] in self.orders]
            ]
        ), 2)

    def cheapest_item(self):
        return (
            [
                y
                for y in self.menu
                if y["price"] == min([x["price"] for x in self.menu])
            ][0]["item"]
            if len(self.menu) > 0
            else 0
        )

    def drinks_only(self):
        return (
            [
                y["item"]
                for y in [x if x["type"] == "drink" else -1 for x in self.menu]
                if y != -1
            ]
            if len(self.menu) > 0
            else []
        )

    def food_only(self):
        return (
            [
                y['item']
                for y in [x if x["type"] == "food" else -1 for x in self.menu]
                if y != -1
            ]
            if len(self.menu) > 0
            else []
        )

def replace_all_punc(astr):
    for x in string.punctuation + string.digits:
        astr = astr.replace(x, '')
    return astr

def compare_strings(astr1, astr2):
    occurences_1 = {}
    occurences_2 = {}
    _ind = 0
    while _ind < len(astr1) or _ind < len(astr2):
        if _ind < len(astr1):
            _1_char = astr1[_ind]
            if _1_char not in occurences_1:
                occurences_1[_1_char] = 1
            else:
                occurences_1[_1_char] += 1
        if _ind < len(astr2):
            _2_char = astr2[_ind]
            if _2_char not in occurences_2:
                occurences_2[_2_char] = 1
            else:
                occurences_2[_2_char] += 1
        _ind += 1
    for eachkey in occurences_1:
        if eachkey not in occurences_2:
            return False
        else:
            if occurences_2[eachkey] < occurences_1[eachkey]:
                return False
    return True
	
        
def hidden_anagram(astr: str, astr2: str):
    astr2 = replace_all_punc(astr2.lower().replace(' ', ''))
    astr = replace_all_punc(astr.lower().replace(' ', ''))
    for i in range(0, len(astr) - len(astr2) + 1):
        _end = i + len(astr2)
        substr = astr[i:_end]

        if compare_strings(substr, astr2):
            ## found string
            return replace_all_punc(astr[i:_end])
    return 'noutfond'

def sort_by_length(arr):
    return sorted(arr, key=len)

def friend(arr):
    return [x for x in arr if len(x) == 4]

def abundant_number(num):
    return sum([x for x in range(1, num) if num % x == 0]) >= num

def area_or_perimeter(l, w):
    return l * w if l == w else (l*2) + (w*2)

def robber_encode(sentence):
    consonant = 'bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ'
    return ''.join(['{}{}{}'.format(x, 'O' if x == x.upper() else 'o', x) if x in consonant else x for x in list(sentence)])

def cakes(recipe, available):
    count = 0
    found_end = False
    commonalities = [x for x in available if x in recipe]
    if len(commonalities) == 0 or len(commonalities) != len([x for x in recipe]):
        return 0
    while not found_end:
        for eachkey in available:
            if eachkey in recipe and recipe[eachkey] > available[eachkey]:
                return count
            elif eachkey in recipe:
                available[eachkey] -= recipe[eachkey]
        count += 1
    return 0


def main():
    recipe = {"apples": 3, "flour": 300, "sugar": 150, "milk": 100, "oil": 100}
    available = {"sugar": 500, "flour": 2000, "milk": 2000}
    print(cakes(recipe, available))

if __name__ == "__main__":
    main()
