#!/usr/bin/env python

__author__ = "WhyKiki"
__version__ = "1.0.0"


import pytz
import datetime


def newMoonStart(tz_utc: object, tz_cet: object, meanLunationPeriod_half: datetime) -> dict:

    ## 1st new moon per definition (Brown, 1900, 2000)
    dict_moons = {}
    dict_moons["newMoon_brown_utc"] = datetime.datetime(1923, 1, 17, 2, 41, 0, tzinfo=tz_utc)
    dict_moons["newMoon_1900_utc"] = datetime.datetime(1900, 1, 1, 13, 51, 0, tzinfo=tz_utc)
    dict_moons["newMoon_2000_utc"] = datetime.datetime(2000, 1, 6, 18, 14, 0, tzinfo=tz_utc)

    ## 1st new moon per definition (Brown, 1900, 2000): convert datetime to CET
    dict_moons["newMoon_brown_cet"] = dict_moons["newMoon_brown_utc"].astimezone(tz_cet)
    dict_moons["newMoon_1900_cet"] = dict_moons["newMoon_1900_utc"].astimezone(tz_cet)
    dict_moons["newMoon_2000_cet"] = dict_moons["newMoon_2000_utc"].astimezone(tz_cet)

    ## 1st full moon (Brown, 1900, 2000)
    dict_moons["fullMoon_brown_utc"] = dict_moons["newMoon_brown_utc"] + meanLunationPeriod_half
    dict_moons["fullMoon_1900_utc"] = dict_moons["newMoon_1900_utc"] + meanLunationPeriod_half
    dict_moons["fullMoon_2000_utc"] = dict_moons["newMoon_2000_utc"] + meanLunationPeriod_half

    ## 1st full moon (Brown, 1900, 2000): convert datetime to CET
    dict_moons["fullMoon_brown_cet"] = dict_moons["fullMoon_brown_utc"].astimezone(tz_cet)
    dict_moons["fullMoon_1900_cet"] = dict_moons["fullMoon_1900_utc"].astimezone(tz_cet)
    dict_moons["fullMoon_2000_cet"] = dict_moons["fullMoon_2000_utc"].astimezone(tz_cet)

    return dict_moons


def evaluateMoons(newMoon_start: datetime, endDate: datetime, meanLunationPeriod: datetime) -> list:
    """Calculate all new moons from start date"""
    newMoon = newMoon_start
    all_newMoons = [newMoon]
    while newMoon < endDate:
        newMoon += meanLunationPeriod
        all_newMoons.append(newMoon)
    return all_newMoons


def main():

    ## define timezones
    tz_utc = pytz.timezone("UTC")
    tz_cet = pytz.timezone("Europe/Berlin")

    ## datetime of now
    now_utc = datetime.datetime.now(tz_utc)
    now_cet = datetime.datetime.now(tz_cet)

    ## mean lunation period (29 days, 12 hours, 43 minutes)
    meanLunationPeriod = datetime.timedelta(days=29, hours=12, minutes=43)
    meanLunationPeriod_half = meanLunationPeriod / 2

    ## define start of first new moon (UTC) according to different definitions and convert to CET
    dict_moons = newMoonStart(tz_utc, tz_cet, meanLunationPeriod_half)

    ## from start of first new moon, calculate all new moons till end date
    lst_allNewMoons = evaluateMoons(newMoon_start=dict_moons["newMoon_2000_cet"],
                                    endDate=now_cet,
                                    meanLunationPeriod=meanLunationPeriod)

    ## from start of first full moon, calculate all full moons till end date
    lst_allFullMoons = evaluateMoons(newMoon_start=dict_moons["fullMoon_2000_cet"],
                                     endDate=now_cet,
                                     meanLunationPeriod=meanLunationPeriod)

    for elem in lst_allFullMoons:
        print(elem)


if __name__ == "__main__":
    main()
