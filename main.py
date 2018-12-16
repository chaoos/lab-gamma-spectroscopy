# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy import interpolate
from scipy.optimize import curve_fit
from scipy.special import erf, erfc
from scipy.interpolate import UnivariateSpline
from scipy.optimize import fsolve
from scipy.integrate import simps, quad
from decimal import Decimal
import pandas as pd
from uncertainties import ufloat, unumpy
from uncertainties.umath import *
from subprocess import call
from datetime import date, datetime
import os
import helpers as hp
import sys
import re

#from simplegeneric import generic

#font = {'family' : 'normal',
#        'size'   :  12}
#matplotlib.rc('font', **font)

def simpson(f, a, b, N):
    """Zusammengesetzte Simpsonregel in 1d.

    Input:
           f : Funktion f(x) welche integiert werden soll.
        a, b : untere/obere Grenze des Integrals.
           N : Anzahl Teilintervalle in der zusammengesetzten Regel.
    """
    x, h = np.linspace(a, b, N+1, retstep=True)
    xm = .5*(x[1:] + x[:-1])
    return h/6.0 * (f(a) + 4.0*sum(f(m) for m in xm) + 2.0*sum(f(z) for z in x[1:-1]) + f(b))

run = {
	'dose rate': 		True,
	'system linearity':	True, # + relative resoluton
	'spectra': 			True,
	'activity': 		True,
	'efficiency': 		True,
	'eu152':			True,
	'activities':		True,
	'efficiencies': 	True,
	'compile': 			True,
}

# Custom code starts here
hp.replace("Name", "Roman Gruber")
hp.replace("Experiment", "Gamma-Spectroscopy")
figNr = 0

a_err = [1, 3]
detectors = [
	{
		'name': 'NaI',
		'file': 'data/data.txt'
	},
	{
		'name': 'HPGe',
		'file': 'data/data_ged.txt'
	}
]

sources = {
	'cs137': {
		'name': r'$^{137}Cs$',
		'file': 'data/cs137',
		'logy': False,
		'xmin': 50,
		'xmax': 800,
		'ymin': 0,
		'ymax': 1.2,
		'keV': False,
		'markers': {
			'Pb x-ray peak': (80.0, 1),
			'backscatter peak': (200.0, 0.6),
			'Compton edge': (460.0, 0.6),
			'661.62 keV': (662.0, 1.1),
		},
	},
	'eu152': {
		'name': r'$^{152}Eu$',
		'file': 'data/eu152',
		'logy': True,
		'xmin': 0,
		'xmax': 1800,
		'ymin': 10**-3,
		'ymax': 10,
		'markers': {
			'Pb x-ray peak': (80.0, 7),
			'121.78 keV': (121, 4),
			'244.67 keV': (244, 2),
			'344.30 keV': (344, 1),
			'778.90 keV': (778, 0.6),
			'964.00 keV': (964, 0.35),
			'1085.80 keV, 1112.07 keV': (1112, 0.2),
			'1408.08 keV': (1408, 0.06),
		},
	},
	'co60': {
		'name': r'$^{60}Co$',
		'file': 'data/co60',
		'xmin': 0,
		'xmax': 1800,
		'ymin': 10**-3,
		'ymax': 1,
		'logy': True,
		'markers': {
			'Pb x-ray peak': (80.0, 0.4),
			'backscatter peak': (220.0, 0.2),
			'Compton edge': (900.0, 0.3),
			'1173.23 keV': (1200, 0.2),
			'1332.51 keV': (1340, 0.1),
		},
	},
	'bi207': {
		'name': r'$^{207}Bi$',
		'file': 'data/bi207',
		'logy': True,
		'xmin': 0,
		'xmax': 2200,
		'ymin': 10**-3,
		'ymax': 100,
		'markers': {
			'Pb x-ray peak': (80.0, 13),
			'backscatter peak': (190.0, 6),
			'Compton edge': (825.0, 1.1),
			'569.67 keV': (570, 2),
			'1063.62 keV': (1063, 0.7),
			'1770.22 keV': (1770, 0.1),
		},
	},
	'na22': {
		'xmin': 0,
		'xmax': 1800,
		'ymin': 10**-3,
		'ymax': 1,
		'name': r'$^{22}Na$',
		'file': 'data/na22',
		'logy': True,
		'markers': {
			'Pb x-ray peak': (80.0, 0.2),
			'backscatter peak': (200.0, 0.1),
			'1274.53 keV': (1274.53, 0.02),
		},
	},
	'gemisch_a': {
		'xmin': 0,
		'xmax': 1800,
		'ymin': 10**-5,
		'ymax': 1,
		'name': r'Gemisch A',
		'file': 'data/gemisch_a',
		'logy': True,
		'markers': {
			'Pb x-ray peak': (80.0, 0.7),
			'backscatter peak': (200.0, 0.4),
			'Compton edge': (460.0, 0.2),
			'661.62 keV (A)': (662.0, 0.1),
		},
	},
	'gemisch_b': {
		'xmin': 0,
		'xmax': 1800,
		'ymin': 10**-4,
		'ymax': 100,
		'name': r'Gemisch B',
		'file': 'data/gemisch_b',
		'logy': True,
		'markers': {
			'Pb x-ray peak': (80.0, 50),
			'backscatter peak': (200.0, 20),
			'Compton edge': (460.0, 12),
			'661.62 keV (A)': (662.0, 7),
		},
	},
	'cs137_ged': {
		'name': r'$^{137}Cs$',
		'name1': r'$^{137}Cs \, at \, 1\, cm$',
		'name4': r'$^{137}Cs \, at \, 4\, cm$',
		'name8': r'$^{137}Cs \, at \, 8\, cm$',
		'file': 'data/cs137_ged',
		'file1': 'data/cs137_ged_1cm',
		'file4': 'data/cs137_ged_4cm',
		'file8': 'data/cs137_ged_8cm',
		'logy': True,
		'markers': {
			'Pb x-ray peak': (80.0, 10),
			'backscatter peak': (200.0, 5),
			'Compton edge': 460.0,
			'661.62 keV': 662.0,
		},
		'ymin': 10**-2,
		'ymax': 10**3,
		'xmin': 0,
		'xmax': 800
	},
	'eu152_ged': {
		'name': r'$^{152}Eu$',
		'name1': r'$^{152}Eu \, at \, 1\, cm$',
		'name4': r'$^{152}Eu \, at \, 4\, cm$',
		'name8': r'$^{152}Eu \, at \, 8\, cm$',
		'file': 'data/eu152_ged',
		'file1': 'data/eu152_ged_1cm',
		'file4': 'data/eu152_ged_4cm',
		'file8': 'data/eu152_ged_8cm',
		'logy': True,
		'markers': {
			'Pb x-ray peak': (30.0, 500),
			'121.78 keV': (121, 300),
			'244.67 keV': (244, 100),
			'344.30 keV': (344, 50),
			'444.00 keV': (444, 5),
			'778.90 keV': (778, 110),
			'867.39 keV': (867, 65),
			'964.00 keV': (964, 40),
			'1085.80 keV': (1085, 25),
			'1112.07 keV': (1112, 10),
			'1408.08 keV': (1408, 5),
		},
		'xmin': 0,
		'xmax': 1800
	},
	'co60_ged': {
		'xmin': 0,
		'xmax': 1800,
		'ymin': 10**-3,
		'ymax': 10,
		'name': r'$^{60}Co$',
		'name1': r'$^{60}Co \, at \, 1\, cm$',
		'name4': r'$^{60}Co \, at \, 4\, cm$',
		'name8': r'$^{60}Co \, at \, 8\, cm$',
		'file': 'data/co60_ged',
		'file1': 'data/co60_ged_1cm',
		'file4': 'data/co60_ged_4cm',
		'file8': 'data/co60_ged_8cm',
		'logy': True,
		'markers': {
			'Pb x-ray peak': (80.0, 2),
			'backscatter peak': (220.0, 0.8),
			'Compton edge': (950.0, 0.6),
			'1173.23 keV': (1173, 7),
			'1332.51 keV': (1332, 5),
		},
	},
	'bi207_ged': {
		'xmin': 0,
		'xmax': 2200,
		'ymin': 10**-3,
		'ymax': 1000,
		'name': r'$^{207}Bi$',
		'name1': r'$^{207}Bi \, at \, 1\, cm$',
		'name4': r'$^{207}Bi \, at \, 4\, cm$',
		'name8': r'$^{207}Bi \, at \, 8\, cm$',
		'file': 'data/bi207_ged',
		'file1': 'data/bi207_ged_1cm',
		'file4': 'data/bi207_ged_4cm',
		'file8': 'data/bi207_ged_8cm',
		'logy': True,
		'markers': {
			'Pb x-ray peak': (80.0, 650),
			'backscatter peak': (190.0, 400),
			'Compton edge': (850.0, 90),
			'569.67 keV': (570, 150),
			'1063.62 keV': (1063, 50),
			'1770.22 keV': (1770, 3),
		},
	},
	'na22_ged': {
		'xmin': 0,
		'xmax': 1800,
		'ymin': 10**-3,
		'ymax': 10,
		'name': r'$^{22}Na$',
		'name1': r'$^{22}Na \, at \, 1\, cm$',
		'name4': r'$^{22}Na \, at \, 4\, cm$',
		'name8': r'$^{22}Na \, at \, 8\, cm$',
		'file': 'data/na22_ged',
		'file1': 'data/na22_ged_1cm',
		'file4': 'data/na22_ged_4cm',
		'file8': 'data/na22_ged_8cm',
		'logy': True,
		'markers': {
			'Pb x-ray peak': (80.0, 6),
			'backscatter peak': (200.0, 3),
			'1274.53 keV': (1274.53, 1),
		},
	},
	'gemisch_a_ged': {
		'xmin': 0,
		'xmax': 1800,
		'ymin': 10**-3,
		'ymax': 1000,
		'name': r'Gemisch A',
		'name1': r'Gemisch A at 1 cm',
		'name4': r'Gemisch A at 4 cm',
		'name8': r'Gemisch A at 8 cm',
		'file': 'data/gemisch_a_ged',
		'file1': 'data/gemischA_ged_1cm',
		'file4': 'data/gemischA_ged_4cm',
		'file8': 'data/gemischA_ged_8cm',
		'logy': True,
		'markers': {
			'Pb x-ray peak': (80.0, 40),
			'backscatter peak': (200.0, 20),
			'Compton edge': (460.0, 10),
			'661.62 keV (A)': (662.0, 500),
			'1173.23 keV (B)': (1173, 0.6),
			'1332.51 keV (C)': (1332, 0.3),
		},
	},
	'gemisch_b_ged': {
		'xmin': 0,
		'xmax': 1800,
		'ymin': 10**-3,
		'ymax': 1000,
		'name': r'Gemisch B',
		'name1': r'Gemisch B at 1 cm',
		'name4': r'Gemisch B at 4 cm',
		'name8': r'Gemisch B at 8 cm',
		'file': 'data/gemisch_b_ged',
		'file1': 'data/gemischB_ged_1cm',
		'file4': 'data/gemischB_ged_4cm',
		'file8': 'data/gemischB_ged_8cm',
		'logy': True,
		'markers': {
			'Pb x-ray peak': (80.0, 50),
			'backscatter peak': (200.0, 20),
			'Compton edge': (460.0, 10),
			'661.62 keV (A)': (662.0, 500),
			'1173.23 keV (B)': (1173, 0.8),
			'1332.51 keV (C)': (1332, 0.2),
		},
	},
}

# get the measurement times from the .mcd files
for index, source in sources.items():
	if 'file1' in source:
		if 'file0' in source:
			z = zip(['file0', 'file1', 'file4', 'file8'], ['time0', 'time1', 'time4', 'time8'])
		else:
			z = zip(['file1', 'file4', 'file8'], ['time1', 'time4', 'time8'])

		for filei, timei in z:
			fp = open(source[filei] + '.mcd', encoding="utf-8")
			for i, line in enumerate(fp):
				if i == 1:
					match = re.search(r'REALTIME: ([\d,\.]+)', line)
					if match:
						source[timei] = float(match.group(1))
						continue
	else:
		fp = open(source['file'] + '.mcd', encoding="utf-8")
		for i, line in enumerate(fp):
			if i == 1:
				match = re.search(r'REALTIME: ([\d,\.]+)', line)
				if match:
					source['time'] = float(match.group(1))
					continue


def annotate(plt, where, what, X, Y, n=1, log=False, Xtol=20):
	if isinstance(where, tuple):
		whereX, whereY = where[0], where[1]
	else:
		whereX, whereY = where, False
	upper = whereX + Xtol/2
	lower = whereX - Xtol/2
	Xrel = X[np.logical_and(X >= lower, X <= upper)]
	Yrel = Y[np.logical_and(X >= lower, X <= upper)]
	y = np.max(Yrel)
	x = Xrel[np.argmax(Yrel)]
	
	if log:
		exp = len(str(int(y))) - 1
		offset = 5*10**exp
	else:
		offset = (np.abs(np.min(Y) - np.max(Y))/10)

	if whereY == False:
		whereY = y+n*offset

	plt.annotate(what, xy=(x, y), xytext=(x, whereY), arrowprops=dict(
		facecolor='black',
		width=1,
		headwidth=5,
		shrink=0.05
	))

	return (x, y)

def peak_yvalue (source, det, energy, Xtol=20):
	channels = np.linspace(0, 8191, 8192).astype(int)

	if det == "NaI":
		ctps = hp.fetch2(sources[source]['file'] + '.txt', 1)/sources[source]['time']
		keV = f_NaI(channels)
	elif det == "HPGe":
		ctps = hp.fetch2(sources[source]['file1'] + '.txt', 1)/sources[source]['time1']
		keV = f_HPGe(channels)

	upper = energy + Xtol/2
	lower = energy - Xtol/2

	Yrel = ctps[np.logical_and(keV >= lower, keV <= upper)]
	return np.max(Yrel)

def exp_param(x, a, b, c, d):
   return ( np.vectorize( lambda x, a, b, c, d: a*np.exp(b*x + c) + d )(x, a, b, c, d) )

# get the keV scale
#ch_NaI = [0.00, 1518.00, 2864.00, 1728.00, 2092.00, 827.00, 2341.00, 2756.00, 1311.00, 2276.00, 3444.00, 2671.00]
#keV_NaI = [0.000000, 661.620000, 1408.080000, 778.900000, 964.000000, 344.300000, 1112.070000, 1332.510000, 569.670000, 1063.620000, 1770.220000, 1274.530000]
ch_NaI = [0.00, 1518.00]
keV_NaI = [0.000000, 661.620000]
keV_HPGe = [0.000000, 661.620000, 1332.510000, 1173.230000, 1408.080000, 1112.070000, 1085.800000, 964.000000, 867.390000, 778.900000, 444.000000, 411.090000, 344.300000, 244.670000, 121.780000, 1770.220000]

ch_HPGe = [0.00, 3006.00, 6073.00, 5345.00, 6418.00, 5065.00, 4943.00, 4389.00, 3948.00, 3541.00, 2012.00, 1860.00, 1557.00, 1102.00, 540.00, 8069.00]
coeffs_NaI = np.polyfit(ch_NaI, keV_NaI, 1)
coeffs_HPGe = np.polyfit(ch_HPGe, keV_HPGe, 1)
f_NaI = lambda x: np.polyval(coeffs_NaI, x)
f_HPGe = lambda x: np.polyval(coeffs_HPGe, x)

deltaC = 1
deltaE_NaI = coeffs_NaI[0]*deltaC
deltaE_HPGe = coeffs_HPGe[0]*deltaC

hp.replace("dE_gamma_NaI", deltaE_NaI)
hp.replace("dE_gamma_HPGe", deltaE_HPGe)

hp.replace("linear_translation_NaI", hp.fmt_fit(coeffs_NaI))
hp.replace("linear_translation_HPGe", hp.fmt_fit(coeffs_HPGe))


if run['eu152']:

	Es_NaI = []
	Es_HPGe = []
	peaks = [121.78, 244.67, 344.30, 444.00, 778.90, 867.39, 964.00, 1085.80, 1112.07, 1408.08]
	for E in peaks:
		Es_NaI.append(hp.physical(peak_yvalue('eu152', 'NaI', E), 0, 3))
		Es_HPGe.append(hp.physical(peak_yvalue('eu152_ged', 'HPGe', E), 0, 3))

	Es_HPGe = np.array(Es_HPGe)
	Es_NaI = np.array(Es_NaI)

	peaks_str = [r'Pb x-ray peak', r'$121.78$ keV', r'$244.67$ keV', r'$344.30$ keV', r'$444.00$ keV', r'$778.90$ keV', r'$867.39$ keV', r'$964.00$ keV', r'$1085.80$ keV', r'$1112.07$ keV', r'$1408.08$ keV']

	ratio = Es_NaI/Es_HPGe

	peaks_str = ['121.78', '244.67', '344.30', '444.00', '778.90', '867.39', '964.00', '1085.80', '1112.07', '1408.08']
	arr = hp.to_table(
		r'Peaks $[keV]$', peaks_str,
		r'$c_1$ $[s^{-1}]$', Es_NaI,
		r'$c_2$ $[s^{-1}]$', Es_HPGe,
		r'ratio $c_1/c_2$', ratio,
	)

	hp.replace("table:eu152:peaks", arr)

	y1 = peak_yvalue('eu152', 'NaI', 1408.08)
	y2 = peak_yvalue('eu152_ged', 'HPGe', 1408.08)

	hp.replace("eu_NaI_peak_y", hp.physical(peak_yvalue('eu152', 'NaI', 1408.08), 0, 3))
	hp.replace("eu_HPGe_peak_y", hp.physical(peak_yvalue('eu152_ged', 'HPGe', 1408.08), 0, 3))

if run['dose rate']:
	print("calculating dose rate ...")

	x = hp.fetch2('data/doserate.xlsx', 'x [cm]', 1)/100
	A = hp.fetch2('data/doserate.xlsx', 'A [muSv/hr]', 0.2)

	A[10].sf = 1

	def invsq_param(x, a):
		return ( np.vectorize( lambda x, a: a/(x**2))(x, a) )

	fit_x = hp.nominal(x)
	fit_x[0] = 10**-10 # "very" small => "zero"
	guesses = np.array([0.0207])
	popt, pcov = curve_fit(invsq_param, fit_x, hp.nominal(A), p0=guesses, sigma=hp.stddev(A), absolute_sigma=True)
	aerr = list(np.sqrt(np.diag(pcov)))
	#a = list(popt)
	a = popt[0]

	invsq_fit = lambda x: np.vectorize( lambda x: 0.02/(x**2) )(x)
	X = np.linspace(0.001, 2.2, 500)

	plt.figure(figNr)
	plt.errorbar(hp.nominal(x*100), hp.nominal(A), fmt='x', xerr=hp.stddev(x*100), yerr=hp.stddev(A), label=r'dose rate data')
	plt.plot(X*100, invsq_fit(X), '-', label=r'inverse square fit function')

	plt.xlabel(r"Distance $[cm]$")
	plt.ylabel(r"Dose $[\mu Sv / hr]$")
	plt.xlim(-10, 120)
	plt.ylim(-0.5, 5)
	plt.legend(loc=1)
	plt.grid(True)
	plt.savefig('plots/dose_rate.eps')
	figNr += 1

	# parameters of the exp fit
	hp.replace("fit:a", hp.physical(0.0207, 0, 3))

	arr = hp.to_table(
		r'$Distance \, [cm]$', x*100,
		r'$Dose \, [\mu S / hr]$', A,
	)

	hp.replace("table:doserate", arr)


# Plots for the spectra
if run['spectra']:
	print("processing plots from all sources ...")
	for index, source in sources.items():

		channels = np.linspace(0, 8191, 8192).astype(int)

		plt.figure(figNr)

		# HPGe detector spectra
		if 'file1' in source:
			keV = f_HPGe(channels)
			ctps1 = hp.fetch2(source['file1'] + '.txt', 1)/source['time1']
			ctps4 = hp.fetch2(source['file4'] + '.txt', 1)/source['time4']
			ctps8 = hp.fetch2(source['file8'] + '.txt', 1)/source['time8']
			if (source['logy'] == True):
				plt.semilogy(keV, ctps1, '.', label=r'source ' + source['name1'])
				plt.semilogy(keV, ctps4, '.', label=r'source ' + source['name4'])
				plt.semilogy(keV, ctps8, '.', label=r'source ' + source['name8'])
			else:
				plt.plot(keV, ctps1, '.', label=r'source ' + source['name1'])
				plt.plot(keV, ctps4, '.', label=r'source ' + source['name4'])
				plt.plot(keV, ctps8, '.', label=r'source ' + source['name8'])
		
		# NaI detector spectra			
		else:
			ctps = hp.fetch2(source['file'] + '.txt', 1)/source['time']
			if 'keV' in source and source['keV'] == False:
				keV = f_NaI(channels)
			else:
				keV = hp.fetch2(source['file'] + '.txt', 0)
			if (source['logy'] == True):
				plt.semilogy(keV, ctps, '.', label=r'source ' + source['name'])
			else:
				plt.plot(keV, ctps, '.', label=r'source ' + source['name'])

		if 'markers' in source:
			Y = ctps1 if 'file1' in source else ctps
			n = 1
			for what, where in source['markers'].items():
				annotate(plt, where, what, keV, Y, n, source['logy'])
				n = n + 1

		if 'ymin' in source:
			plt.ylim(source['ymin'], source['ymax'])

		if 'xmin' in source:
			plt.xlim(source['xmin'], source['xmax'])

		plt.ylabel(r"Counts per second $[s^{-1}]$")
		plt.xlabel(r"Energy $[keV]$")
		plt.grid(True)
		plt.savefig('plots/' +  os.path.basename(source['file']) + '.eps')

		figNr += 1

	#plt.show()
	#exit()

# Plot of linearity of the measurement system
if run['system linearity']:

	i = 1
	for det in detectors:
		print("processing " + det['name'] + "-detector data ...")

		dE_gamma = np.abs(hp.fetch2(det['file'], 'dC-pos [keV]'))
		dE_channel = np.abs(hp.fetch2(det['file'], 'dC-pos [ch]'))
		E_gamma = hp.fetch2(det['file'], 'C-pos [keV]', dE_gamma)
		E_channel = hp.fetch2(det['file'], 'C-pos [ch]', dE_channel)

		coeffs = hp.phpolyfit(E_gamma, E_channel, 1)
		p = lambda x: np.polyval(coeffs, x)
		pu = lambda x: np.polyval(coeffs.copy() + hp.stddev(coeffs), x)
		pl = lambda x: np.polyval(coeffs.copy() - hp.stddev(coeffs), x)
		x = np.linspace(0, 2500, 5)

		E_gamma_cs = E_gamma[0]
		E_channel_cs = E_channel[0]
		E_gamma_eu = E_gamma[1:6]
		E_channel_eu = E_channel[1:6]
		E_gamma_co = E_gamma[6:8]
		E_channel_co = E_channel[6:8]
		E_gamma_na = E_gamma[8]
		E_channel_na = E_channel[8]
		E_gamma_bi = E_gamma[9:]
		E_channel_bi = E_channel[9:]

		plt.figure(figNr)
		figNr += 1
		plt.errorbar(hp.nominal(E_gamma_cs), hp.nominal(E_channel_cs), fmt='x', xerr=hp.stddev(E_gamma_cs), yerr=hp.stddev(E_channel_cs), label=r'$^{137}Cs$')
		plt.errorbar(hp.nominal(E_gamma_eu), hp.nominal(E_channel_eu), fmt='x', xerr=hp.stddev(E_gamma_eu), yerr=hp.stddev(E_channel_eu), label=r'$^{152}Eu$')
		plt.errorbar(hp.nominal(E_gamma_co), hp.nominal(E_channel_co), fmt='x', xerr=hp.stddev(E_gamma_co), yerr=hp.stddev(E_channel_co), label=r'$^{60}Co$')
		plt.errorbar(hp.nominal(E_gamma_na), hp.nominal(E_channel_na), fmt='x', xerr=hp.stddev(E_gamma_na), yerr=hp.stddev(E_channel_na), label=r'$^{22}Na$')
		plt.errorbar(hp.nominal(E_gamma_bi), hp.nominal(E_channel_bi), fmt='x', xerr=hp.stddev(E_gamma_bi), yerr=hp.stddev(E_channel_bi), label=r'$^{207}Bi$')

		plt.plot(hp.nominal(x), hp.nominal(p(x)), '-r', label=r'linear fit $p_{' + det['name'] + r'}(x)$')
		plt.plot(hp.nominal(x), hp.nominal(pu(x)), 'g--', linewidth=0.5, label=r'error of the linear fit')
		plt.plot(hp.nominal(x), hp.nominal(pl(x)), 'g--', linewidth=0.5)
		plt.fill_between(hp.nominal(x), hp.nominal(pu(x)), hp.nominal(pl(x)), facecolor="lightyellow", alpha=0.5, linewidth=0.0)

		plt.ylabel(r"$\gamma-Energy \, [channels]$")
		plt.xlabel(r"$\gamma-Energy \, [keV]$")
		plt.xlim(0, 1900)
		plt.ylim(bottom=0)
		plt.grid(True)
		plt.legend(loc="best")
		plt.savefig('plots/system_linearity_' + det['name'] + '.eps')

		hp.replace("linear_fit_" + det['name'], hp.fmt_fit(coeffs))
		hp.replace("linear_fit_" + det['name'] + ':b', coeffs[1])
		hp.replace("linear_fit_" + det['name'] + ':a_err', hp.physical(coeffs[0].s, 0, a_err[i-1]))

		# Plot of relative resolution power dE/E against E
		dfwhm = np.abs(hp.fetch2(det['file'], 'dFWHM [keV]'))
		fwhm = hp.fetch2(det['file'], 'FWHM [keV]', dfwhm)
		dEdE = fwhm/E_gamma*100

		dEdE_cs = dEdE[0]
		dEdE_eu = dEdE[1:6]
		dEdE_co = dEdE[6:8]
		dEdE_na = dEdE[8]
		dEdE_bi = dEdE[9:]

		guesses = np.array([3.0, -0.05, 0.0, 0.0])
		popt, pcov = curve_fit(exp_param, hp.nominal(E_gamma), hp.nominal(dEdE), p0=guesses, sigma=hp.stddev(dEdE), absolute_sigma=True)
		aerr, berr, cerr, derr = list(np.sqrt(np.diag(pcov)))
		a, b, c, d = list(popt)
		exp_fit2 = lambda x: np.vectorize( lambda x: a*np.exp(b*x + c) + d )(x)
		X = np.linspace(0.01, 1800, 500)

		plt.figure(figNr)
		figNr += 1
		plt.plot(X, exp_fit2(X), '-', label=r'exponential fit function $f_{' + det['name'] + r'}(x)$')
		plt.errorbar(hp.nominal(E_gamma_cs), hp.nominal(dEdE_cs), fmt='x', xerr=hp.stddev(E_gamma_cs), yerr=hp.stddev(dEdE_cs), label=r'$^{137}Cs$')
		plt.errorbar(hp.nominal(E_gamma_eu), hp.nominal(dEdE_eu), fmt='x', xerr=hp.stddev(E_gamma_eu), yerr=hp.stddev(dEdE_eu), label=r'$^{152}Eu$')
		plt.errorbar(hp.nominal(E_gamma_co), hp.nominal(dEdE_co), fmt='x', xerr=hp.stddev(E_gamma_co), yerr=hp.stddev(dEdE_co), label=r'$^{60}Co$')
		plt.errorbar(hp.nominal(E_gamma_na), hp.nominal(dEdE_na), fmt='x', xerr=hp.stddev(E_gamma_na), yerr=hp.stddev(dEdE_na), label=r'$^{22}Na$')
		plt.errorbar(hp.nominal(E_gamma_bi), hp.nominal(dEdE_bi), fmt='x', xerr=hp.stddev(E_gamma_bi), yerr=hp.stddev(dEdE_bi), label=r'$^{207}Bi$')

		plt.ylabel(r"$\Delta E / E_{\gamma} [\%]$")
		plt.xlabel(r"$E_{\gamma} \, [keV]$")
		plt.grid(True)
		plt.legend(loc="best")
		plt.savefig('plots/relative_resolution_' + det['name'] + '.eps')

		hp.replace("det:" + str(i), i)
		hp.replace("a_det:" + str(i), hp.physical(a, 0, 3))
		hp.replace("b_det:" + str(i), hp.physical(b, 0, 3))
		hp.replace("c_det:" + str(i), hp.physical(c, 0, 3))
		hp.replace("d_det:" + str(i), hp.physical(d, 0, 3))
		i = i + 1

# Activity according to the theory
if run['activity']:
	A_then = hp.physical(42.8 * 10**3, 0, 3) # Bq @ 1.4.2000
	t_half = hp.physical(30.1*365*86400, 0, 3) # 30.1 years
	elapsed_time = (datetime.now() - datetime(2000, 4, 1)).days*86400 # time in sec since 1.4.200
	tau = t_half/np.log(2)
	A_now = A_then*hp.pnumpy.exp(-elapsed_time/tau)
	hp.replace("A_then", A_then*10**-3)
	hp.replace("A_now", A_now*10**-3)


	# Activity from the spectra of Cs137
	N_photo = hp.physical(86.8243, 0.60, 4) # from data.txt: Area[ctps] of Cs137
	emission_prob = hp.physical(84.62/100, 0, 4) # from A.4 intensity of the 661.62 keV peak of Cs137

	# A.3 solid angle correction
	a_0 = 35*10**-3 # 35 mm
	r_d = 0.5*31.75*10**-3 # 1/2 * 31.75 mm
	a_d = 25.4*10**-3 # 24.5 mm
	r_0 = 10*10**-3 # 10 mm

	s_0 = lambda r: a_d / np.sqrt((r/a_0)**2 + 1)
	s_1 = lambda r: (r/a_0*(r_d - r)) / np.sqrt((r/a_0)**2 + 1)
	s_2 = lambda r: (r/a_0*(r - r_0)) / np.sqrt((r/a_0)**2 + 1)

	mu_NaI = 0.4*10**2
	mu_Pb = 1.5*10**2
	eps_NaI = hp.physical(0.42, 0.05, 2)

	hp.replace("mu_NaI", mu_NaI)
	hp.replace("mu_Pb", mu_Pb)
	hp.replace("eps_NaI", eps_NaI)

	f1 = lambda r: 2*np.pi/(a_0**2) * r * (1 - hp.pnumpy.exp(-mu_NaI*s_0(r)))
	f2 = lambda r: 2*np.pi/(a_0**2) * r * (1 - hp.pnumpy.exp(-mu_NaI*s_1(r))) * np.exp(-mu_Pb*s_2(r))

	omega_eps1 = quad(f1, 0, r_0)
	omega_eps2 = quad(f2, 0, r_0)
	omega_eps = omega_eps1[0] + omega_eps2[0]

	omega_D = omega_eps/eps_NaI
	hp.replace("omega_D", omega_D)

	Delta = hp.physical(0.3, 0.05, 4)

	A_meas = N_photo*4*np.pi/(omega_eps*Delta*emission_prob)

	Delta.sf = 2
	dE_gamma = np.abs(hp.fetch2(detectors[0]['file'], 'dC-pos [keV]'))[0]
	E_gamma_cs = hp.fetch2(detectors[0]['file'], 'C-pos [keV]', dE_gamma)[0]
	hp.replace("A_meas", A_meas*10**-3)
	hp.replace("E_gamma_cs", E_gamma_cs)
	hp.replace("Delta", Delta)
	hp.replace("emission_prob", emission_prob)
	hp.replace("omega_eps", omega_eps)

# Efficiency
if run['efficiency']:
	#https://www.hindawi.com/journals/stni/2014/186798/
	#It is defined as the ratio of the number of counts recorded by the detector () to the number of radiation () emitted by the source (in all directions)

	A = np.pi*r_d**2
	r = np.sqrt(r_d**2 + a_0**2)
	omega_D = A/r**2
	#print(A/a_0**2)
	#print(omega_D)
	N_calc = A_now*omega_D/(4*np.pi)
	N_det = N_photo/emission_prob
	efficiency = (N_det/N_calc)*100 # in percent

	hp.replace("efficiency", efficiency)


if run['activities']:
	print("processing activity of the sources for the NaI detector ...")

	# Table A.4 in the instruction sheet
	prob = {
		'cs137_661.62': 0.8462,
		#'eu152_121.78': 0.2924,
		#'eu152_244.67': 0.07616,
		'eu152_344.30': 0.2700,
		#'eu152_444.00': 0.02839,
		'eu152_778.90': 0.1299,
		#'eu152_867.39': 0.04176,
		'eu152_964.00': 0.1458,
		#'eu152_1085.80': 0.1029,
		'eu152_1112.07': 0.1358,
		'eu152_1408.08': 0.2121,
		'bi207_569.67': 0.98,
		'bi207_1063.62': 0.77,
		'bi207_1770.22': 0.07,
		'na22_1274.53': 0.99944,
		'co60_1173.23': 0.9986,
		'co60_1332.51': 0.9998
	}

	# Figure A2 in the instruction sheet
	deltas_NaI = {
		'cs137_661.62': hp.physical(0.3, 0.05, 4),
		#'eu152_121.78': 0.83,
		#'eu152_244.67': 0.7,
		'eu152_344.30': hp.physical(0.6, 0.05, 4),
		#'eu152_444.00': 0.55,
		'eu152_778.90': hp.physical(0.28, 0.05, 4),
		#'eu152_867.39': 0.22,
		'eu152_964.00': hp.physical(0.19, 0.05, 4),
		#'eu152_1085.80': 0.16,
		'eu152_1112.07': hp.physical(0.14, 0.05, 4),
		'eu152_1408.08': hp.physical(0.11, 0.05, 4),
		'bi207_569.67': hp.physical(0.34, 0.05, 4),
		'bi207_1063.62': hp.physical(0.15, 0.05, 4),
		'bi207_1770.22': hp.physical(0.1, 0.05, 4),
		'na22_1274.53': hp.physical(0.13, 0.05, 4),
		'co60_1173.23': hp.physical(0.15, 0.05, 4),
		'co60_1332.51': hp.physical(0.11, 0.05, 4)
	}

	# Figure A3 in the instruction sheet
	mus_NaI = {
		'cs137_661.62': 0.4*10**2,
		#'eu152_121.78': 4.5*10**2,
		#'eu152_244.67': 0.95*10**2,
		'eu152_344.30': 0.6*10**2,
		#'eu152_444.00': 0.42*10**2,
		'eu152_778.90': 0.3*10**2,
		#'eu152_867.39': 0.3*10**2,
		'eu152_964.00': 0.23*10**2,
		#'eu152_1085.80': 0.21*10**2,
		'eu152_1112.07': 0.2*10**2,
		'eu152_1408.08': 0.19*10**2,
		'bi207_569.67': 0.35*10**2,
		'bi207_1063.62': 0.21*10**2,
		'bi207_1770.22': 0.18*10**2,
		'na22_1274.53': 0.2*10**2,
		'co60_1173.23': 0.2*10**2,
		'co60_1332.51': 0.19*10**2
	}

	# Figure A4 in the instruction sheet
	mus_Pb = {
		'cs137_661.62': 1.5*10**2,
		#'eu152_121.78': 49*10**2,
		#'eu152_244.67': 6.5*10**2,
		'eu152_344.30': 3.5*10**2,
		#'eu152_444.00': 2.2*10**2,
		'eu152_778.90': 0.85*10**2,
		#'eu152_867.39': 0.84*10**2,
		'eu152_964.00': 0.8*10**2,
		#'eu152_1085.80': 0.75*10**2,
		'eu152_1112.07': 0.75*10**2,
		'eu152_1408.08': 0.65*10**2,
		'bi207_569.67': 0.55*10**2,
		'bi207_1063.62': 0.8*10**2,
		'bi207_1770.22': 0.55*10**2,
		'na22_1274.53': 0.7*10**2,
		'co60_1173.23': 0.74*10**2,
		'co60_1332.51': 0.68*10**2
	}

	strings = np.array([
		r'$^{137}Cs$',
		#r'$^{152}Eu$',
		#r'$^{152}Eu$',
		r'$^{152}Eu$',
		#r'$^{152}Eu$',
		r'$^{152}Eu$',
		#r'$^{152}Eu$',
		r'$^{152}Eu$',
		#r'$^{152}Eu$',
		r'$^{152}Eu$',
		r'$^{152}Eu$',
		r'$^{207}Bi$',
		r'$^{207}Bi$',
		r'$^{207}Bi$',
		r'$^{22}Na$',
		r'$^{60}Co$',
		r'$^{60}Co$'
	])

	energies = np.array([
		hp.physical(661.62, 0, 5),
		#121.78,
		#244.67,
		hp.physical(344.30, 0, 5),
		#444.00,
		hp.physical(778.90, 0, 5),
		#867.39,
		hp.physical(964.00, 0, 5),
		#1085.80,
		hp.physical(1112.07, 0, 6),
		hp.physical(1408.08, 0, 6),
		hp.physical(569.67, 0, 5),
		hp.physical(1063.62, 0, 6),
		hp.physical(1770.22, 0, 6),
		hp.physical(1274.53, 0, 6),
		hp.physical(1173.23, 0, 6),
		hp.physical(1332.51, 0, 6)
	])

	# A.3 solid angle correction
	a_0 = 35*10**-3 # 35 mm
	r_d = 0.5*31.75*10**-3 # 1/2 * 31.75 mm
	a_d = 25.4*10**-3 # 24.5 mm
	r_0 = 10*10**-3 # 10 mm

	s_0 = lambda r: a_d / np.sqrt((r/a_0)**2 + 1)
	s_1 = lambda r: (r/a_0*(r_d - r)) / np.sqrt((r/a_0)**2 + 1)
	s_2 = lambda r: (r/a_0*(r - r_0)) / np.sqrt((r/a_0)**2 + 1)
	
	area_ctps = hp.fetch2(detectors[0]['file'], 'Area [ct/s]')
	df = pd.read_csv(detectors[0]['file'], sep=None, engine='python', header='infer')
	comment = np.array(df['Comment'])

	A_meas = []
	for N_photo, comm in zip(area_ctps, comment):
		emission_prob = prob[comm]
		Delta = deltas_NaI[comm]
		mu_NaI = mus_NaI[comm]
		mu_Pb = mus_Pb[comm]

		f1, f2 = None, None
		f1 = lambda r: 2*np.pi/(a_0**2) * r * (1 - hp.pnumpy.exp(-mu_NaI*s_0(r)))
		f2 = lambda r: 2*np.pi/(a_0**2) * r * (1 - hp.pnumpy.exp(-mu_NaI*s_1(r))) * np.exp(-mu_Pb*s_2(r))

		omega_eps1 = quad(f1, 0, r_0)
		omega_eps2 = quad(f2, 0, r_0)
		omega_eps = omega_eps1[0] + omega_eps2[0]

		A_meas.append(N_photo*4*np.pi/(omega_eps*Delta*emission_prob))

	A_meas = np.array(A_meas)

	arr = hp.to_table(
		r'Source', strings,
		r'Peak $[keV]$', energies,
		r'Activity $[kBq]$', A_meas/1000,
	)

	hp.replace("table:activities", arr)

	Cs_y = peak_yvalue('cs137_ged', 'HPGe', 662.61)
	Cs_A = A_meas[0]
	gemischA_y1 = peak_yvalue('gemisch_a_ged', 'HPGe', 662.61)
	gemischA_Cs_A = gemischA_y1*Cs_A/Cs_y
	gemischB_y1 = peak_yvalue('gemisch_b_ged', 'HPGe', 662.61)
	gemischB_Cs_A = gemischB_y1*Cs_A/Cs_y

	Co_y = peak_yvalue('co60_ged', 'HPGe', 1173.23)
	Co_A = A_meas[10]
	gemischA_y2 = peak_yvalue('gemisch_a_ged', 'HPGe', 1173.23)
	gemischA_Co_A = gemischA_y2*Co_A/Co_y
	gemischB_y2 = peak_yvalue('gemisch_b_ged', 'HPGe', 1173.23)
	gemischB_Co_A = gemischB_y2*Co_A/Co_y

	arr = hp.to_table(
		r'Source', np.array(['Gemisch A', 'Gemisch B']),
		r'$^{137}Cs$ Activity [kBq]', np.array([gemischA_Cs_A, gemischB_Cs_A])/1000,
		r'$^{60}Co$ Activity [Bq]', np.array([gemischA_Co_A, gemischB_Co_A])
	)

	hp.replace("tab:unknownActivites", arr)

# Efficiency
if run['efficiencies']:
	print("processing efficiencies of the peaks for the NaI detector ...")
	#https://www.hindawi.com/journals/stni/2014/186798/
	#It is defined as the ratio of the number of counts recorded by the detector () to the number of radiation () emitted by the source (in all directions)

	A = np.pi*r_d**2
	r = np.sqrt(r_d**2 + a_0**2)
	omega_D = A/r**2

	# Table A.4 in the instruction sheet
	emission_probs = np.array([
		0.8462,
		0.2700,
		0.1299,
		0.1458,
		0.1358,
		0.2121,
		0.98,
		0.77,
		0.07,
		0.99944,
		0.9986,
		0.9998
	])

	A_nows = np.array([
		A_meas[0],
		np.sum(A_meas[1:6])/(0.2700 + 0.1299 + 0.1458 + 0.1358 + 0.2121),
		np.sum(A_meas[1:6])/(0.2700 + 0.1299 + 0.1458 + 0.1358 + 0.2121),
		np.sum(A_meas[1:6])/(0.2700 + 0.1299 + 0.1458 + 0.1358 + 0.2121),
		np.sum(A_meas[1:6])/(0.2700 + 0.1299 + 0.1458 + 0.1358 + 0.2121),
		np.sum(A_meas[1:6])/(0.2700 + 0.1299 + 0.1458 + 0.1358 + 0.2121),
		np.sum(A_meas[6:9])/(0.98),
		np.sum(A_meas[6:9])/(0.77 + 0.07),
		np.sum(A_meas[6:9])/(0.77 + 0.07),
		A_meas[9]/0.99944,
		np.sum(A_meas[10:])/0.9986,
		np.sum(A_meas[10:])/0.9998
	])

	area_ctps = hp.fetch2(detectors[0]['file'], 'Area [ct/s]')

	efficiencies = []
	for A_now, emission_prob, N_photo in zip(A_nows, emission_probs, area_ctps):
		N_calc = A_now*omega_D/(4*np.pi)
		N_det = N_photo/emission_prob
		efficiencies.append((N_det/N_calc)*100)

	efficiencies = np.array(efficiencies)
	eff = hp.pharray(hp.nominal(efficiencies), hp.stddev(efficiencies), 4, None, None, 3)

	arr = hp.to_table(
		r'Source', strings,
		r'Peak $[keV]$', energies,
		r'Efficiency $[\%]$', eff,
	)

	hp.replace("table:efficiencies", arr)

if run['compile']:
	hp.compile()