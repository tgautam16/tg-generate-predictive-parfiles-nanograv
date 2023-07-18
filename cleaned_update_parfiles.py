#!/usr/bin/env python
# coding: utf-8

import os, os.path, glob, pickle
import numpy as np
import pint.toa as pt
import pint.models as pm
import pint.fitter as pf
import pint.residuals as pr
from pint.fitter import ConvergenceFailure
import pint.logging
import timing_analysis.utils as utils
from timing_analysis.timingconfiguration import TimingConfiguration
from astropy import log
import astropy.units as au
import matplotlib.pyplot as plt
from astropy import units as u
from astropy.time import Time
pint.logging.setup(level="ERROR")
plt.rcParams.update({'font.size': 22})
#Changing working directory to the base of timing analysis folder
os.chdir('../.')

class Pred_par:
    # Need configfile to initiate the class
    def __init__(self,psr,configfile,MJDlow=57000.0*au.d,MJDhigh=58000.0*au.d,SNRcut=8.0,zthres=1.0): 
        self.configfile=configfile
        self.psr=psr
        self.MJDlow=MJDlow
        self.MJDhigh=MJDhigh
        self.SNRcut=SNRcut
        self.zthres=zthres
        tc=TimingConfiguration(self.configfile)
        self.tc = tc
        
    # Load the original 15yr TOAs and model
    def load_toas(self):
        mo,to = self.tc.get_model_and_toas(excised=True,usepickle=True)
        if not to: mo,to = self.tc.get_model_and_toas(apply_initial_cuts=True)
        return mo,to
    
    # Apply manual cuts
    def cut_toas(self,to):
        self.tc.manual_cuts(to)
    
    # Read and merge recent TOAs 
    def load_merge_qck_toas(self,tqfilenms,quicklookdir,to,mo):
        # Read in all the recent quicklook TOAs
        tq = pt.get_TOAs(tqfilenms, model=mo) 
        # Remove all quicklook TOAs below our SNR threshold
        snrs = np.asarray([float(snr) for snr in tq.get_flag_value("snr")[0]])
        tq.table = tq.table[snrs > self.SNRcut]
        # Combine the TOAs
        tall = pt.merge_TOAs([to, tq])
        return tall
    
    # Remove all lines that begin with the following
        # FD params, FB derivs (but not FB0), all DMX params, and red noise params
        # and for TEMPO compatability, also remove SWM, troposphere, and 
        # planet shapiro correction.  And finally, remove ECORR.
        # also remove JUMP statement that correspond to an old receiver not making the MJD cut
        
    def simplify_parfile(self, tall, parout,otherflag=None):
        """Convert a complicated 15yr parfile into a simplified parfile"""
        rcvr=np.unique(tall.get_flag_value("fe")[0])
        #rcvr=np.unique(tall.table['fe'])
        a=tall.table['fe','mjd']
        for rc in rcvr:
            x=str(rc)==a["fe"]
            mjdcut = Time(self.MJDlow, format='mjd', scale='utc')
            maxmjd=(max(a["mjd"][x]))
            if((max(a["mjd"][x])) < mjdcut):
                print("Reciever jump must be removed from model: ", rc)
                otherflag=f"fe {rc}"
        
        parin = self.tc.get_model_path()
        remove = ["FD", "FB", "DMX", "RN", "SWM", "CORRECT", "PLANET", "INFO", "DMDATA", "NE_SW", "ECORR"]
        with open(parin, "r") as fi, open(parout, "w") as fo:
            for line in fi:
                if line.startswith("CLOCK"):
                    line = line.replace("CLOCK", "CLK")
                    line = line.replace("BIPM2019", "BIPM")
                if otherflag is not None:
                    if otherflag in line:
                        line = None
                        break
                for param in remove:
                    if line.startswith(param):
                        if not line.startswith("FB0"):
                            line = None
                            break
                if line is not None:
                    fo.write(line)
        ms = pm.get_model(parout)
        return ms  
    
    # Freeze/fix all the fitted parameters 
    def freeze_fit_pars(self,ms): 
        for param in ms.free_params:
            getattr(ms, param).frozen = True
        # Explicitly fit for F0, DM, and JUMPs
        getattr(ms, "F0").frozen = False
        getattr(ms, "DM").frozen = False
        getattr(ms, "JUMP1").frozen = False
        if hasattr(ms, "JUMP2"):
            getattr(ms, "JUMP2").frozen = False
        if hasattr(ms, "JUMP3"):
            getattr(ms, "JUMP3").frozen = False
        # If this is a black widow, fit for TASC and FB0
        if hasattr(ms, "FB0") and ms.FB0.value is not None:
            getattr(ms, "FB0").frozen = False
            getattr(ms, "TASC").frozen = False
        print("Free params:", ms.free_params)
        
    # Fit TOAs using a simple WLS fitter    
    def fit_TOAs(self,tall,ms,iterations):
        f = pf.DownhillWLSFitter(tall, ms)
        try:
            f.fit_toas(maxiter=iterations)
            f.model.CHI2.value = f.resids.chi2
        except ConvergenceFailure:
            log.warning('Failed to converge; moving on with best result.')
        print("Postfit RMS = ", f.resids.rms_weighted())
        return f
    
    # Use z-score to remove outliers
    def cut_outliers(self,f):
        stddev= np.std(f.resids.time_resids)
        mean=np.mean(f.resids.time_resids) 
        zscore=np.fabs(((f.resids.time_resids-mean)/stddev).decompose())
        zcut=zscore<self.zthres
        N0 = tall.ntoas
        tall.table = tall.table[zcut]
        print(f"Removing {N0-zcut.sum()} TOAs of {N0} as outliers")



run_15yr = True
print(os.getcwd())
# Set TOA and config file locations:
quicklookdir = "/nanograv/timing/pipeline/quicklook/data/vegas/"
configs = glob.glob("/nanograv/share/15yr/timing/intermediate/20230628.Release.nbwb.ce0b6e7e/narrowband/config/*.nb.yaml")
#configs=glob.glob("/home/jovyan/work/timing_analysis/configs/*.nb.yaml")
configs = [x for x in configs if ("gbt" not in x and "ao" not in x)]
psrs = {os.path.split(x)[1].split(".")[0] for x in configs}
niter = 3 #Iterations for fitting
#Input range of data-set (*For 15 yr data-release, adding last three years only*)
MJDlow = 57976 *au.d  #11Aug,2017
MJDhigh = 59071 *au.d  #10Aug,2020
if not os.path.exists('release_tools/pred_par/'):
    os.mkdir('release_tools/pred_par/')
print("Pulsars for which predictive par files are being generated: ", psrs)
for psr in psrs:
    print(f"Working on {psr}")
    configfile = [x for x in configs if psr in x][0]
    print(configfile)
    # intantiate the class and load TOAs
    t=Pred_par(psr,configfile,MJDlow,MJDhigh) 
    mo,to = t.load_toas()
    # Apply manual cuts
    t.cut_toas(to)
    if run_15yr == False:
        # Load and merge quicklook TOAs
        tqfilenms = glob.glob(f"{quicklookdir}/{psr}/{psr}*.quicklook.x.nb.tim")
        if len(tqfilenms)==0:
            log.error(f"No quicklook TOAs for {psr}.  ")
            tall = to
        else:
            tall = t.load_merge_qck_toas(tqfilenms,quicklookdir,to,mo)
    else:     
        tall = to
    tall.table = tall.table[:]
    tall.compute_pulse_numbers(mo)
    # Create an initial simple model
    ms = t.simplify_parfile(tall, f"release_tools/pred_par/{psr}_simple.par")
    t.freeze_fit_pars(ms)
    # Remove TOAs outside the range provided
    tall.table = tall.table[t.MJDlow < tall.get_mjds()]
    tall.table = tall.table[t.MJDhigh > tall.get_mjds()]
    # Calculate the pulse numbers based on the 15yr solutions
    tall.compute_pulse_numbers(mo)
    # Use a simple WLS fitter
    f=t.fit_TOAs(tall,ms,niter)
#     f.plot()
    t.cut_outliers(f)
    f=t.fit_TOAs(tall,ms,niter)
#     f.plot()
    # Write the new parfile and remove simple ones to avoid confusion
    f.model.CLOCK.value = "TT(BIPM)"
    f.model.write_parfile(f"release_tools/pred_par/{psr}_pred.par", format="tempo")
    if os.path.exists(f"release_tools/pred_par/{psr}_simple.par"):
        os.remove(f"release_tools/pred_par/{psr}_simple.par")
    
    
