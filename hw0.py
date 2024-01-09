"""
Cynthia Hong
CSE 163 AF
This program implements of the function of HW0
"""


def funky_sum(a, b, mix):
    """
    Gives three numbers a, b, mix
    returns a if mix smaller or equal to zero
    returns b if mix greater or equal to ones
    returns the sum of the difference between one and mix times a
    and b times mix, if mix is between zero and one
    """
    if mix <= 0:
        return a
    elif mix >= 1:
        return b
    else:
        return (1 - mix) * a + mix * b


def total(n):
    """
    Gives number n
    if n is smaller than zero, returns None
    if n bigger or equal to zero, returns the sum of integers from zero to n
    """
    if n < 0:
        return None
    else:
        result = 0
        for i in range(n + 1):
            result += i
        return result


def swip_swap(source, c1, c2):
    """
    Gives a string source and characters c1 and c2
    returns a string with all occurrences of c1 and c2 swapped in the source
    if source do not include c1 or c2, return source itself
    """
    result = ""
    for letter in source:
        if letter == c1:
            result += c2
        elif letter == c2:
            result += c1
        else:
            result += letter
    return result
