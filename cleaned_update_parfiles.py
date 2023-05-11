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



class TOA_tg:
    
    def __init__(self,psr,configfile,MJDlow=59000.0 *au.d,SNRcut=8.0,zthres=1.0): #Initial configfile to initiate the class
        self.configfile=configfile
        self.psr=psr
        self.MJDlow=MJDlow
        self.SNRcut=SNRcut
        self.zthres=zthres
        tc=TimingConfiguration(self.configfile)
        self.tc = tc
    
    def load_toas(self):
        # Load the original 15yr TOAs and model
        mo,to = self.tc.get_model_and_toas(excised=True,usepickle=True)
        if not to: mo,to = self.tc.get_model_and_toas(apply_initial_cuts=True)
        return mo,to
    
    def cut_toas(self,to):
        self.tc.manual_cuts(to)
        
    def load_merge_qck_toas(self,tqfilenms,quicklookdir,to,mo):
        # Read in all the recent quicklook TOAs
        tq = pt.get_TOAs(tqfilenms, model=mo) 
        # Remove all quicklook TOAs below our SNR threshold
        snrs = np.asarray([float(snr) for snr in tq.get_flag_value("snr")[0]])
        tq.table = tq.table[snrs > self.SNRcut]
        # Combine the TOAs
        tall = pt.merge_TOAs([to, tq])
        tall.table = tall.table[:]
        # Calculate the pulse numbers based on the 15yr solutions
        tall.compute_pulse_numbers(mo)
        #def remove_qck_toas(self):
        return tall
    
    def simplify_parfile(self, tall, parout,otherflag=None):
        """Convert a complicated 15yr parfile into a simplified parfile"""
        # We will remove all lines that begin with the following
        # FD params, FB derivs (but not FB0), all DMX params, and red noise params
        # and for TEMPO compatability, also remove SWM, troposphere, and 
        # planet shapiro correction.  And finally, remove ECORR.
        # Remove JUMP statement that correspond to an old receiver - Find reciever and remove jump from simple model if no mjd more than MJDlow
        rcvr=np.unique(tall.table['fe'])
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
    
    def freeze_fit_pars(self,ms):
            # Freeze/fix all the fitted parameters  
        for param in ms.free_params:
            getattr(ms, param).frozen = True
        # Explicitly fit for F0, DM, and JUMP1 (should check if needed)
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
        
        
    def fit_TOAs(self,tall,ms,iterations):
            # Use a simple WLS fitter
        f = pf.DownhillWLSFitter(tall, ms)
        try:
            f.fit_toas(maxiter=iterations)
            f.model.CHI2.value = f.resids.chi2
        except ConvergenceFailure:
            log.warning('Failed to converge; moving on with best result.')
        print("Postfit RMS = ", f.resids.rms_weighted())
        return f
    
    def cut_outliers(self,f):
            #using z-score approach to remove outliers:
        stddev= np.std(f.resids.time_resids)
        mean=np.mean(f.resids.time_resids) #the ideal scenario of 0 mean is reasonable here
        #mean=0
        zscore=np.fabs(((f.resids.time_resids-mean)/stddev).decompose())
        zcut=zscore<self.zthres
        N0 = tall.ntoas
        tall.table = tall.table[zcut]
        print(f"Removing {N0-zcut.sum()} TOAs of {N0} as outliers")



# Initial parameter set:
#configs=glob.glob("/home/jovyan/work/timing_tg/update_parfiles/*.nb.yaml")
quicklookdir = "/nanograv/timing/pipeline/quicklook/data/vegas/"
configs=glob.glob("/home/jovyan/work/timing_analysis/configs/*.nb.yaml")
configs = [x for x in configs if ("gbt" not in x and "ao" not in x)]
psrs = {os.path.split(x)[1].split(".")[0] for x in configs}
print(psrs)
for psr in psrs:
    print(f"Working on {psr}")
    configfile = [x for x in configs if psr in x][0]
    t=TOA_tg(psr,configfile) # intantiate the class
    mo,to = t.load_toas()
    t.cut_toas(to)
    # J1705-1903 is drifting so bad we need a special starting model
    if psr=="J1705-1903":
        mo = pm.get_model("J1705-1903_start.par")
    tqfilenms = glob.glob(f"{quicklookdir}/{psr}/{psr}*.quicklook.x.nb.tim")
    if len(tqfilenms)==0:
        log.error(f"No quicklook TOAs for {psr}.  Skipping.")
        continue
    tall = t.load_merge_qck_toas(tqfilenms,quicklookdir,to,mo)
    ms = t.simplify_parfile(tall, f"simple_ephem_tg/{psr}_simple.par")
    t.freeze_fit_pars(ms)
    # Remove all of the early TOAs
    tall.table = tall.table[t.MJDlow < tall.get_mjds()]
    # Calculate the pulse numbers based on the 15yr solutions
    tall.compute_pulse_numbers(mo)
    # Use a simple WLS fitter
    f=t.fit_TOAs(tall,ms,2)
    f.plot()
    t.cut_outliers(f)
    f=t.fit_TOAs(tall,ms,2)
    f.plot()
    # Write the new parfile
    f.model.CLOCK.value = "TT(BIPM)"
    f.model.write_parfile(f"simple_ephem_tg/{psr}_fold.par", format="tempo")
    


#psrs with issue: 

#No quicklook TOAs in VEGAS:

#J1312+0051 (no quicklook TOAs)
#J0709+0458 (no quicklook TOAs)
#B1953+29 (no quicklook TOAs)
#J1453+1902 (no quicklook TOAs)

#J1614-2230 (one bad observation)
#1719-1438 (Last obs, too bad!)
#2043+1711 (too few TOAs?)



#*1705-1903* (- try multiple iterations)
#*1713+0747* (sudden offset and drift from there) -- good timing, ignore
#*B1937+21* (Huge trend in the TOAs at the beginning) -- dmx trend, ignore




