#!/usr/bin/env python
# encoding: utf-8
#
# VW_PCA.py
#
# Created by Kate Rowlands on 4th May 2017.

import numpy as np
import math
from scipy.io.idl import readsav
from astropy.io import ascii
import matplotlib.patches as patches
from matplotlib.path import Path
from marvin.tools.maps import Maps
from marvin.utils.dap.bpt import get_masked
from marvin.utils.dap.bpt import get_snr

class VW_PCA():

    def SFR(self, maps, distance):
        #calculate log10(SFR) from the Halpha flux, following Kennicutt et al. (1998).
        # Assumes a Salpeter IMF.

        snr = 3.

        hb = get_masked(maps, 'hb_4862', snr=get_snr(snr, 'hb'))
        ha = get_masked(maps, 'ha_6564', snr=get_snr(snr, 'ha'))

        # Av derivation from A_Ha following Jorge from CALIFA paper (Catalán-Torrecilla et al. (2015)
        K_Ha = 2.53
        K_Hb = 3.61

        balmerdec = np.log10((ha/hb)/2.86)
        A_Ha = balmerdec * K_Ha/(-0.4*(K_Ha-K_Hb))
        Ha_cor = ha*(10.**(0.4*A_Ha))

        Lha = Ha_cor*1e-17*4.*math.pi*((3.0857e18*1E6*distance)**2)
        K98 = 7.9*1e-42
        SFR = np.log10(K98*Lha)

        return SFR

    def dustlaw_gal(self, wave, flux, ebvgal):
        #Galactic dust correction. Based on dustlaw_gal.pro by Vivienne Wild.

        data = ascii.read("cardelli_gal.ext")

        lam_dust = data['col1']
        dust = data['col2']

        if np.amin(wave) < np.amin(lam_dust):
            print 'wavelength wrong'
        if np.amax(wave) > np.amax(lam_dust):
            print 'wavelength wrong'

        f_dust = np.interp(wave, lam_dust, dust) #interpolate

        R = 3.1                         #R_V
        A = ebvgal*(f_dust + R)           #extinction in mags

        #now we know what the extinction is as function of lambda, so divide
        #this out of the flux
        flux_corr = flux/10**(-A/2.5)    #corrected flux
        corr = 10**(-A/2.5)              #to correct error array if needed

        return corr

    def masksky(self, wave, flux):
        # masks sky emission lines [OI], N2+. Based on masksky.pro witten by Vivienne Wild

        emlines = [[5574,5590],[4276,4282],[6297,6305],[6364,6368]]
        nlines = np.shape(emlines)[0]

        n = 0
        index2 = [0] * 1000
        mask_arr = np.zeros(1000)

        for i in range(nlines):

            index = np.where((wave > emlines[i][0]) & (wave < emlines[i][1]))
            count = len(wave[index])

            if count > 0:
                ind_mask = np.where(((wave > emlines[i][0]-75) & (wave < emlines[i][0])) | \
                                   ((wave > emlines[i][1]) & (wave < emlines[i][1]+75)))
                mask_arr[n:n+count] = np.median(flux[ind_mask])
                index2[n:n+count] = np.squeeze(index)
                n = n + count

        if n != 0:
            index2_new = np.array(index2[0:n])
            index2_new.astype(int)

            pixel = np.zeros(len(wave))
            pixel.astype(int)
            pixel[index2_new] = 1

            newflux=flux
            newflux[index2_new] = mask_arr[0:n]

        if n == 0:
            if n_elements(silent) == 0:
                print 'masksky: no em lines masked'
                newflux = flux
                pixel = np.zeros(len(wave))

        return newflux, pixel

    def pca_prepro(self, waverest, z, flux, error, ebvgal):
        #Does preprocessing of MaNGA spectra needed before doing PCA, e.g.
        #Dust correction, skyline masking and emission line masking.

        espec = readsav('/data/kate/StAndrews/PCA_GAMA_PSB_stacking/VO/PCARUN/ESPEC/pcavo_espec_25.sav')
        npix = len(espec.wave)

        waveobs = waverest*(1.+z)

        #Dust correct
        corr = self.dustlaw_gal(waveobs, flux, ebvgal)
        fluxdustcorr = flux/corr
        errordustcorr = error/corr

        fluxdustcorr_new, pixel = self.masksky(waveobs, fluxdustcorr)

        skymask = np.where(pixel == 1)
        errordustcorr[skymask] = 0.

        #Masking
        masksize = 5.0
        indmask = np.where((waverest > 4102.9-masksize) & (waverest < 4102.9+masksize))
        errordustcorr[indmask]=0.
        indmask = np.where((waverest > 3971.195-masksize) & (waverest < 3971.195+masksize))
        errordustcorr[indmask]=0.
        indmask = np.where((waverest > 3890.151-masksize) & (waverest < 3890.151+masksize))
        errordustcorr[indmask]=0.

        masksize = 2.5
        indmask = np.where((waverest > 3836.472-masksize) & (waverest < 3836.472+masksize))
        errordustcorr[indmask]=0.
        indmask = np.where((waverest > 3798.976-masksize) & (waverest < 3798.976+masksize))
        errordustcorr[indmask]=0.

        indbad = np.where(np.isnan(fluxdustcorr_new) == True)
        fluxdustcorr_new[indbad] = 0.
        errordustcorr[indbad] = 0.

        #Interpolate onto SDSS eigenbasis
        newflux = np.interp(espec.wave, waverest, fluxdustcorr_new)
        newerror_sq = np.interp(espec.wave, waverest, errordustcorr**2)

        #bad pixel masks get screwed up by interpolation
        indmask = np.where((np.isnan(newflux) == True) | (newflux == 0.))
        newerror_sq[indmask] = 0.
        newflux[indmask] = 0.
        newerror = np.sqrt(newerror_sq)

        return newflux, newerror

    def PCA_classify(self, points, vertices):
        #Classify spaxels into different PCA classes.

        # SF
        sf_verts = [
            (vertices["sf_vert1"], vertices["junk_y_lower"]), # left, bottom
            (vertices["sf_vert2"], vertices["psb_cut"]), # left, top
            (vertices["sf_vert3"], vertices["psb_cut"]), # right, top
            (vertices["sf_vert4"], vertices["junk_y_lower"]) # right, bottom
            ]

        sf_path = Path(sf_verts)
        SF = sf_path.contains_points(points)

        # PSB
        psb_verts = [
            (vertices["left_cut"], vertices["psb_cut"]), # left, bottom
            (vertices["left_cut"], vertices["junk_y_upper"]), # left, top
            (vertices["green_vert1"]-2., vertices["junk_y_upper"]), # right, top
            (vertices["green_vert1"], vertices["psb_cut"]) # right, bottom
            ]

        psb_path = Path(psb_verts)
        PSB = psb_path.contains_points(points)

        # SB
        sb_verts = [
            (vertices["left_cut"], vertices["sb_vert1"]), # left, bottom
            (vertices["left_cut"], vertices["psb_cut"]), # left, top
            (vertices["sf_vert2"], vertices["psb_cut"]), # right, top
            (vertices["sf_vert1"]-0.2, vertices["sb_vert1"]) # right, bottom
            ]

        sb_path = Path(sb_verts)
        SB = sb_path.contains_points(points)

        # Green valley
        green_verts = [
            (vertices["sf_vert4"], vertices["junk_y_lower"]), # left, bottom
            (vertices["sf_vert3"], vertices["psb_cut"]), # left, top
            (vertices["green_vert1"], vertices["psb_cut"]), # right, top
            (vertices["green_vert2"], vertices["junk_y_lower"]) # right, bottom
            ]

        green_path = Path(green_verts)
        Green = green_path.contains_points(points)

        # Red
        red_verts = [
            (vertices["green_vert2"], vertices["junk_y_lower"]), # left, bottom
            (vertices["green_vert1"]+1.7, vertices["junk_y_upper"]), # left, top
            (vertices["right_cut"], vertices["junk_y_upper"]), # right, top
            (vertices["right_cut"], vertices["junk_y_lower"]) # right, bottom
            ]

        red_path = Path(red_verts)
        Red = red_path.contains_points(points)

        # Junk
        junk_verts = [
            (vertices["sf_vert1"]-0.3, vertices["junk_y_lower2"]), # left, bottom
            (vertices["sf_vert2"]-0.3, vertices["junk_y_lower"]), # left, top
            (vertices["right_cut"], vertices["junk_y_lower"]), # right, top
            (vertices["right_cut"], vertices["junk_y_lower2"]) # right, bottom
            ]

        junk_path = Path(junk_verts)
        Junk = junk_path.contains_points(points)

        class_map = np.empty(len(points[:,0]))
        class_map.fill(-99.)
        class_map[Red] = 1.
        class_map[SF] = 2.
        class_map[SB] = 3.
        class_map[Green] = 4.
        class_map[PSB] = 5.
        class_map[Junk] = 0.
        class_map[np.where(np.add(points[:,0], points[:,1]) == 0.)] = 0.

        return class_map
