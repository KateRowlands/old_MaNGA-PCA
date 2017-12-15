#!/usr/bin/env python
# encoding: utf-8
#
# MaNGA_Utils.py
#
# Created by Kate Rowlands on 14 March 2017.

import numpy as np

def spx_map(binid):
    #Make a mask so we don't double count spaxels which are in a bin (and therefore have the same value)
    
    binids = np.sort(np.unique(binid))
    map = np.zeros(shape=(len(binid),len(binid)))

    #Store positions of (first spaxel of) binids which have one or more spaxels
    #First element of -1 is excluded
    for x in binids[1:]:
        storex, storey = np.where(binid == x)

        #For central spaxel
        middle_spx_x = int(np.median(storex))
        middle_spx_y = int(np.median(storey))
        map[middle_spx_x, middle_spx_y] = 1

    return map
