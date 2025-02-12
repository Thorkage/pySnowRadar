from bisect import bisect
from datetime import datetime

#Adapted from here https://stackoverflow.com/questions/33415475/how-to-get-current-date-and-time-from-gps-unsegment-time-in-python

_LEAP_DATES = ((1981, 6, 30), (1982, 6, 30), (1983, 6, 30),
               (1985, 6, 30), (1987, 12, 31), (1989, 12, 31),
               (1990, 12, 31), (1992, 6, 30), (1993, 6, 30),
               (1994, 6, 30), (1995, 12, 31), (1997, 6, 30),
               (1998, 12, 31), (2005, 12, 31), (2008, 12, 31),
               (2012, 6, 30), (2015, 6, 30), (2016, 12, 31))

# go here to check if any new leap seconds are added:
# ftp://hpiers.obspm.fr/iers/bul/bulc               

LEAP_DATES = tuple(datetime(*i, 23, 59, 59) for i in _LEAP_DATES)

def utcleap(gps_sec):
    '''converts gps time to utc time, accounting for leap years'''
    epoch_date = datetime.utcfromtimestamp(gps_sec)
    return gps_sec - bisect(LEAP_DATES, epoch_date)
