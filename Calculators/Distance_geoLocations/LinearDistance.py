#!/usr/bin/env python

__author__ = "WhyKiki"
__version__ = "1.0.0"


import math
import numpy as np


def degMinSecToDecDeg(degMinSec: str) -> float:
    """
    Convert coordinates (format: degrees minutes seconds) to decimal degree.
    E.g. 07° 36´ 24´´ N    -->   7.6066666666666665
    :param degMinSec: String, coordinate (either latitude or longitude) whose format is to be converted
    :return: float, coordinate in decimal degree
    """
    ## split coordinate into degrees, minutes, seconds
    degMinSec = degMinSec.split(" ")

    ## extract degrees, minutes, seconds
    deg_ = float(degMinSec[0][0:2])
    min_ = float(degMinSec[1][0:2])
    sec_ = float(degMinSec[2][0:2])

    ## convert degrees, minutes and seconds to a single decimal degree number
    dd = deg_ + min_ / 60 + sec_ / (60 * 60)

    ## take directional location into consideration
    direction = degMinSec[3]
    if direction == "S" or direction == "W":
        dd *= -1

    return dd


def decDegToDegMinSec(dd: float) -> list:
    """
    Convert decimal coordinate degree (for either latitude or longitude) to other format (degree minute second).
    E.g.: 7.6066666666666665    -->    [7, 36, 23.99999999999949]
    :param dd: float, decimal coordinate
    :return: list of numerics (int for degree and minute, float for second),
    """
    deg_ = int(dd)
    min_deg_diff = abs(dd - deg_) * 60
    min_ = int(min_deg_diff)
    sec_deg_diff = (min_deg_diff - min_) * 60
    return [deg_, min_, sec_deg_diff]


def degreesToRadians(degrees: str) -> float:
    """
    Convert decimal degree (of either lat or lon) to radians.
    :param degrees: String, coordinate
    :return: float, radians of coordinate
    """
    ## convert degrees, minutes, seconds to decimal degrees
    dd = degMinSecToDecDeg(degrees)

    ## convert to radians
    return dd * math.pi / 180


def roundNumber(num: float) -> float:
    """
    Round number to 2 digits.
    :param num: float, number to be rounded
    :return: float, rounded number to 2 digits
    """
    if (num % 1) >= 0.5:
        return math.ceil(num * 100) / 100
    elif not np.isnan(num):
        return round(num * 100) / 100
    else:
        None

    return num


def distanceBetweenEarthCoordinates(lat_lon_1: list, lat_lon_2: list) -> float:
    """
    Calculate distance between two coordinates.
    :param lat_lon_1: list of floats decimal coordinates [lon, lat] of place 1
    :param lat_lon_2: list of floats decimal coordinates [lon, lat] of place 2
    :return: float, linear distance between coordinates (place1, place 2) in km
    """
    ## define variable for earth radius in km
    earthRadiusKm = 6371

    ## evaluate distances in latitude and longitude (in radians)
    dLat = degreesToRadians(lat_lon_2[1]) - degreesToRadians(lat_lon_1[1])
    dLon = degreesToRadians(lat_lon_2[0]) - degreesToRadians(lat_lon_1[0])

    ## convert latitude degrees to radians
    lat1 = degreesToRadians(lat_lon_1[1])
    lat2 = degreesToRadians(lat_lon_2[1])

    ## take earth curvature into consideration
    a1 = math.sin(dLat / 2) * math.sin(dLat / 2)
    a2 = math.sin(dLon / 2) * math.sin(dLon / 2) * math.cos(lat1) * math.cos(lat2)
    a = a1 + a2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    ## calculate linear distance
    distance_km = roundNumber(earthRadiusKm * c)
    distance_miles = roundNumber(distance_km * 0.621371)

    return distance_km, distance_miles


def main():

    ## Example
    coord1 = ["07° 35´ 01´´ N", "47° 32´ 28´´ E"]
    coord2 = ["07° 36´ 39´´ N", "47° 32´ 05´´ E"]

    ## calculate linear distances between two locations in km
    print("Location 1:", coord1)
    print("Location 2:", coord2)
    distance_km, distance_miles = distanceBetweenEarthCoordinates(coord1, coord2)
    print("Linear Distance [km]:", distance_km)
    print("Linear Distance [mi]:", distance_miles)


if __name__ == "__main__":
    main()
