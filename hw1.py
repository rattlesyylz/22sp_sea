"""
Cynthia Hong
CSE 163 AF
This program implements of the function of HW1.
"""


def total(n):
    """
    Gives n.
    Returns the sum of the numbers from 0 to n (inclusive).
    If n is negative, returns None.
    """
    if n < 0:
        return None
    else:
        result = 0
        for i in range(n + 1):
            result += i
        return result


def count_divisible_digits(n, m):
    """
    Gives n and m.
    Returns the number of digits in n that are divisible by m.
    If m is 0, returns zero.
    Any digit in n that is zero is divisible by any number,
    and m is a non-negative single digit.
    """
    count = 0
    n = abs(n)
    if m == 0:
        return 0
    elif n == 0:
        return 1
    else:
        while n > 0:
            digit = n % 10
            n = n // 10
            if digit % m == 0:
                count += 1
        return count


def is_relatively_prime(n, m):
    """
    Gives n and m.
    Returns ture if n and m are relatively prime.
    Returns false if n and m are not relatively prime.
    Assumes n and m are at least 1.
    n and m share no common factors besides 1,
    meaning they are relatively prime.
    """
    number = 1
    while not(number > n) and not(number > m):
        number += 1
        if n % number == 0 and m % number == 0:
            return False
    return True


def travel(location, x, y):
    """
    Gives location, x, and y.
    Returns the tuple base on the directed movements
    according to the direction of location.
    The upper or lower letter case of the characters in the location
    should be ignored, and any characters that are not 'N', 'E', 'W',
    or 'S' should be ignored.
    """
    for letter in location:
        if letter.upper() == "N":
            y += 1
        elif letter.upper() == "E":
            x += 1
        elif letter.upper() == "S":
            y -= 1
        elif letter.upper() == "W":
            x -= 1
    return x, y


def reformat_date(date, current_date, target_date):
    """
    Gives data, current_date, and target_date.
    Reads the date as the format of the current_date.
    Returns the date format with the order of target_date.
    """
    dic_current = {}
    dic_target = {}
    new_date = date.split('/')
    new_current_date = current_date.split('/')
    new_target_date = target_date.split('/')
    for i in range(len(new_current_date)):
        dic_current[new_current_date[i]] = new_date[i]
    for i in range(len(new_target_date)):
        dic_target[new_target_date[i]] = dic_current[new_target_date[i]]
    return "/".join(dic_target.values())


def longest_word(file_name):
    """
    Gives a string filename.
    Returns the longest word in the file and its line.
    If there is a ties for the longest word, returns the first appear word.
    If the file is empty or there is no words, returns None.
    """
    new_word = ""
    largest_num = 0
    with open(file_name) as file_song:
        lines = file_song.readlines()
        if len(lines) == 0:
            return None
        else:
            for i in range(len(lines)):
                words = lines[i].split()
                for word in words:
                    if len(word) > largest_num:
                        largest_num = len(word)
                        count = i + 1
                        new_word = str(count) + ": " + word
            return new_word


def get_average_in_range(values, low, high):
    """
    Gives value(a list of integers) and integer low and high
    Returns the average of all values that lies
    between low(inclusive) and high(exclusive).
    If there is no values in the range, returns 0.
    """
    count = 0
    total = 0
    average = 0
    for value in values:
        if value >= low and value < high:
            total += value
            count += 1
        if count == 0:
            average = 0
        else:
            average = total / count
    return average


def mode_digit(n):
    """
    Gives n.
    Returns the digit that appears most frequently in the number.
    If there is a tie of most frequent digit, returns the greatest value.
    The digit always be non-negative no mather what the values is.
    """
    n = abs(n)
    counts = {}
    max_digit = 0
    most_time = 0
    while n > 0:
        digit = n % 10
        n = n // 10
        if digit in counts:
            counts[digit] += 1
        else:
            counts[digit] = 1
    for digit, time in counts.items():
        if time >= most_time:
            most_time = time
            max_digit = max(max_digit, digit)
    return max_digit
