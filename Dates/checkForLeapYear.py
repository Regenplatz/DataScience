#!/usr/bin/env python

__author__ = "WhyKiki"
__version__ = "1.0.0"


def checkIfLeapYear(year: int) -> bool:
    """
    Check if a given year is a leap year or not. A leap year is defined as divisible by 4, NOT by 100, but by 400.
    """
    leapYear = False
    if year % 4 == 0:
        leapYear = True
        if year % 100 == 0:
            if year % 400 == 0:
                leapYear = True
            else:
                leapYear = False
    return leapYear


def main():
    checkIfLeapYear(2021)


if __name__ == "__main__":
    main()
