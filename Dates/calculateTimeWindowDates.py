#!/usr/bin/env python

__author__ = "WhyKiki"
__version__ = "1.0.0"


from datetime import date, datetime, timedelta
from dateutil.relativedelta import relativedelta
import calendar
from typing import Tuple


def stringToDate(dateOfInterest: str) -> date:
    """Convert String to Date."""
    return datetime.strptime(dateOfInterest, "%d-%m-%Y").date()


def calculateDates(refDate: date, numMonth: int,
                   refDateMonthIncluded: bool) -> Tuple[date, date]:
    """
    Calculate start date and end date for the time window of interest
    """
    ## last day of month (related to refDate)
    str_lastDayOfMonth = calendar.monthrange(refDate.year, refDate.month)[1]
    str_lastOfMonth = f"{str_lastDayOfMonth}-{refDate.month}-{refDate.year}"
    date_lastOfMonth = stringToDate(str_lastOfMonth)
    date_firstOfNextMonth = date_lastOfMonth + timedelta(days=1)

    ## first day of month (related to refDate)
    str_firstOfMonth = f"1-{refDate.month}-{refDate.year}"
    date_firstOfMonth = stringToDate(str_firstOfMonth)
    date_lastOfPreviousMonth = date_firstOfMonth - timedelta(days=1)

    ## calculate start and end date of time window
    if refDateMonthIncluded is True:
        start_date = date_firstOfNextMonth - relativedelta(months=numMonth)
        # end_date = date_lastOfMonth
        end_date = refDate
    else:
        start_date = date_firstOfMonth - relativedelta(months=numMonth)
        end_date = date_lastOfPreviousMonth

    return start_date, end_date


def main():

    refDate = date.today()
    # refDate = datetime(2018, 6, 4).date()
    start_date, end_date = calculateDates(refDate=refDate,
                                          numMonth=12,
                                          refDateMonthIncluded=True)

    print(f"Reference Date: {refDate}")
    print(f"Time window1 from {start_date} to {end_date}")


if __name__ == "__main__":
    main()
