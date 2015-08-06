#! /usr/bin/env python
"""
Models visibilities for various catalog sources and creates a new Miriad UV
file containing either the simulated data, or the residual when the model
is removed from measured data.

Modified from the same named script of Aaron Parsons by Shifan Zuo.
"""
import numpy as n, aipy as a, optparse, os, sys, shutil, ephem

o = optparse.OptionParser()
o.set_usage('mdlvis.py [options] *.uv')
o.set_description(__doc__)
a.scripting.add_standard_options(o, ant=True, cal=True, src=True)
o.add_option('-m','--mode', dest='mode', default='sim',
    help='Operation mode.  Can be "sim" (output simulated data), "sub" (subtract from input data), or "add" (add to input data).  Default is "sim"')
o.add_option('-f', '--flag', dest='flag', action='store_true',
    help='If outputting a simulated data set, mimic the data flagging of the original dataset.')
#o.add_option('-m', '--map', dest='map',
#    help='The Healpix map to use for simulation input.')
#o.add_option('--iepoch', dest='iepoch', default=ephem.J2000, 
#    help='The epoch of coordinates in the map. Default J2000.')
#o.add_option('--freq', dest='freq', default=.150, type='float',
#    help='Frequency of flux data in map.')
o.add_option('-n', '--noiselev', dest='noiselev', default=0., type='float',
    help='RMS amplitude of noise (Jy) added to each UV sample of simulation.')
o.add_option('--nchan', dest='nchan', default=1024, type='int',
    help='Number of channels in simulated data if no input data to mimic.  Default is 1024')
o.add_option('--sfreq', dest='sfreq', default=0.700, type='float',
    help='Start frequency (GHz) in simulated data if no input data to mimic.  Default is 0.700')
o.add_option('--sdf', dest='sdf', default=0.100/1024, type='float',
    help='Channel spacing (GHz) in simulated data if no input data to mimic.  Default is 0.100/1024')
o.add_option('--inttime', dest='inttime', default=1.0, type='float',
    help='Integration time (s) in simulated data if no input data to mimic.  Default is 1.0')
o.add_option('--startjd', dest='startjd', default=2454600., type='float',
    help='Julian Date to start observation if no input data to mimic.  Default is 2454600')
o.add_option('--endjd', dest='endjd', default=2454601., type='float',
    help='Julian Date to end observation if no input data to mimic.  Default is 2454601')
o.add_option('--pol', dest='pol',
    help='Polarizations to simulate (xx,yy,xy,yx) if starting file from scratch.')
opts, args = o.parse_args(sys.argv[1:])

assert(len(args) > 0), 'No uv file name. Usage: mdlvis.py [options] *.uv'
for filename in args:
    assert filename.endswith('.uv'), 'Incorrect input file name'
if opts.mode == 'sim':
    assert(not opts.flag and not (opts.pol is None)), 'Need polarizations and no flag to simulate from scratch'
    
# Parse command-line options
if not opts.mode == 'sim':
    uv = a.miriad.UV(args[0])
    aa = a.cal.get_aa(opts.cal, uv['sdf'], uv['sfreq'], uv['nchan'])
    p,d,f = uv.read(raw=True)
    no_flags = n.zeros_like(f)
    del(uv)
else:
    aa = a.cal.get_aa(opts.cal, opts.sdf, opts.sfreq, opts.nchan)
    no_data = n.zeros(opts.nchan, dtype=n.complex64)
    no_flags = n.zeros(opts.nchan, dtype=n.int32)

nants = len(aa)

# Generate a model of the sky with point sources and a pixel map

#mfq,a1s,a2s,ths = [], [], [], []
#dras,ddecs = [], []
# Initialize point sources
#if not opts.src is None:
srclist,cutoff,catalogs = a.scripting.parse_srcs(opts.src, opts.cat)
cat = a.cal.get_catalog(opts.cal, srclist, cutoff, catalogs)
mfq = cat.get('mfreq')
a1s,a2s,ths = cat.get('srcshape')
#a1s.append(a1); a2s.append(a2); ths.append(th)
dras, ddecs = cat.get('ionref')
#dras.append(dra); ddecs.append(ddec)

## Initialize pixel map
#if not opts.map is None:
#    h = a.map.Map(fromfits=opts.map)
#    px = n.arange(h.npix())
#    try: mflx, i_poly = h[px]
#    except(ValueError): mflx = h[px]
#    px = n.compress(mflx > 0, px)
#    try: mflx, i_poly = h[px]
#    except(ValueError):
#        mflx = h[px]
#        i_poly = [n.zeros_like(mflx)]
#    mind = i_poly[0]    # Only implementing first index term for now
#    mmfq = opts.freq * n.ones_like(mind)
#    mfq.append(mmfq)
#    x,y,z = h.px2crd(px, ncrd=3)
#    m_eq = n.array((x,y,z))
#    # Should pixels include information for resolving them?
#    a1s.append(n.zeros_like(mmfq))
#    a2s.append(n.zeros_like(mmfq))
#    ths.append(n.zeros_like(mmfq))
#    dras.append(n.zeros_like(mmfq))
#    ddecs.append(n.zeros_like(mmfq))
#mfq = n.concatenate(mfq)
#a1s = n.concatenate(a1s)
#a2s = n.concatenate(a2s)
#ths = n.concatenate(ths)
#dras = n.concatenate(dras)
#ddecs = n.concatenate(ddecs)

# A pipe for applying the model
curtime = None
def mdl(uv, p, d, f):
    global curtime, eqs
    uvw, t, (i,j) = p
    pol = a.miriad.pol2str[uv['pol']]
    # if i == j: return p, d, f
    if curtime != t:
        curtime = t
        aa.set_jultime(t)
        #eqs,flx,ind = [],[],[]
        #if not opts.src is None:
        cat.compute(aa)
        eqs = cat.get_crds('eq', ncrd=3)
        flx = cat.get_jys()
        #if not opts.map is None:
        #    m_precess = a.coord.convert_m('eq','eq',
        #        iepoch=opts.iepoch, oepoch=aa.epoch)
        #    eqs.append(n.dot(m_precess, m_eq))
        #    flx.append(mflx); ind.append(mind)
        #eqs = n.concatenate(eqs, axis=-1)
        #flx = n.concatenate(flx)
        #ind = n.concatenate(ind)
        aa.sim_cache(eqs, flx, mfreqs=mfq, 
            ionrefs=(dras,ddecs), srcshapes=(a1s,a2s,ths))
    aa.set_active_pol(pol)
    sd = aa.sim(i, j)
    if opts.mode.startswith('sim'):
        d = sd
        if not opts.flag: f = no_flags
    elif opts.mode.startswith('sub'):
        d -= sd
    elif opts.mode.startswith('add'):
        d += sd
    else:
        raise ValueError('Mode "%s" not supported.' % opts.mode)
    if opts.noiselev != 0:
        # Add on some noise for a more realistic experience
        noise_amp = n.random.random(d.shape) * opts.noiselev
        noise_phs = n.random.random(d.shape) * 2*n.pi * 1j
        noise = noise_amp * n.exp(noise_phs)
        d += noise * aa.passband(i,j)
    return p, n.where(f, 0, d), f

if not opts.mode == 'sim':
    # Run mdl on all files
    for filename in args:
        uvofile = filename + 's'
        print filename,'->',uvofile
        if os.path.exists(uvofile):
            print 'File exists: skipping'
            continue
        uvi = a.miriad.UV(filename)
        a.scripting.uv_selector(uvi, opts.ant)
        uvo = a.miriad.UV(uvofile, status='new')
        uvo.init_from_uv(uvi)
        uvo.pipe(uvi, mfunc=mdl, raw=True, append2hist="MDLVIS: " + ' '.join(sys.argv) + '\n')
else:
    # Initialize a new UV file
    pols = opts.pol.split(',')
    # uv = a.miriad.UV('new.uv', status='new')
    if os.path.exists(args[0]):
        shutil.rmtree(args[0])
    uv = a.miriad.UV(args[0], status='new')
    uv._wrhd('obstype','mixed-auto-cross')
    uv._wrhd('history','MDLVIS: created file.\nMDLVIS: ' + ' '.join(sys.argv) + '\n')
    uv.add_var('telescop','a'); uv['telescop'] = 'Tianlai 16 dishes'
    uv.add_var('antdiam'   ,'r'); uv['antdiam'] = 6.0 # antenna diameter
    uv.add_var('project','a'); uv['project'] = 'Some project'
    uv.add_var('observer','a'); uv['observer'] = 'Some one'
    uv.add_var('operator','a'); uv['operator'] = 'Some one'
    uv.add_var('version' ,'a'); uv['version'] = '0.0.1'
    uv.add_var('epoch'   ,'r'); uv['epoch'] = 2000.0
    uv.add_var('source'  ,'a'); uv['source'] = 'North Pole'
    uv.add_var('latitud' ,'d'); uv['latitud'] = aa.lat
    uv.add_var('dec'     ,'d'); uv['dec'] = n.pi / 2.0 # declination of the North pole
    uv.add_var('obsdec'  ,'d'); uv['obsdec'] = n.pi / 2.0 # declination of the North pole
    uv.add_var('longitu' ,'d'); uv['longitu'] = aa.long
    uv.add_var('pntdec' ,'d'); uv['pntdec'] = aa.lat # declination of the pointing center
    uv.add_var('pntra' ,'d'); uv['pntra'] = aa.long # declination of the pointing center
    uv.add_var('npol'    ,'i'); uv['npol'] = len(pols)
    uv.add_var('nspect'  ,'i'); uv['nspect'] = 1
    uv.add_var('nants'   ,'i'); uv['nants'] = nants
    uv.add_var('antpos'  ,'d')
    antpos = n.array([ant.pos for ant in aa], dtype=n.double)
    uv['antpos'] = antpos.transpose().flatten()
    uv.add_var('sfreq'   ,'d'); uv['sfreq'] = opts.sfreq
    uv.add_var('freq'    ,'d'); uv['freq'] = opts.sfreq
    uv.add_var('restfreq','d'); uv['restfreq'] = opts.sfreq
    uv.add_var('sdf'     ,'d'); uv['sdf'] = opts.sdf
    uv.add_var('nchan'   ,'i'); uv['nchan'] = opts.nchan
    uv.add_var('nschan'  ,'i'); uv['nschan'] = opts.nchan
    uv.add_var('inttime' ,'r'); uv['inttime'] = float(opts.inttime)
    # These variables just set to dummy values
    uv.add_var('vsource' ,'r'); uv['vsource'] = 0.
    uv.add_var('ischan'  ,'i'); uv['ischan'] = 1
    uv.add_var('tscale'  ,'r'); uv['tscale'] = 0.
    uv.add_var('veldop'  ,'r'); uv['veldop'] = 0.
    uv.add_var('pressmb'  ,'r'); uv['pressmb'] = 0. # atmospheric pressure, millibar
    uv.add_var('systemp'  ,'r'); uv['systemp'] = 0. # Antenna system temperatures, Kelvin
    # These variables will get updated every spectrum
    uv.add_var('coord'   ,'d')
    uv.add_var('time'    ,'d')
    uv.add_var('lst'     ,'d')
    uv.add_var('ra'      ,'d')
    uv.add_var('obsra'   ,'d')
    uv.add_var('baseline','r')
    uv.add_var('pol'     ,'i')

    # Now start generating data
    times = n.arange(opts.startjd, opts.endjd, opts.inttime/a.const.s_per_day)
    for cnt,t in enumerate(times):
        print 'Timestep %d / %d' % (cnt+1, len(times))
        aa.set_jultime(t)
        uv['lst'] = aa.sidereal_time()
        uv['ra'] = aa.sidereal_time()
        uv['obsra'] = aa.sidereal_time()

        # choose baselines
        ants = a.scripting.parse_ants(opts.ant, nants)
        ants = list(set(ants)) # unique the list
        bls = [] # baseline pairs
        if not ants:
            # ants is an empty list
            bls = [(i, j) for i in range(nants) for j in range(i, nants)] # all baseline pairs
        elif len(ants) == 1 and ants[0][0] == 'auto':
            if ants[0][1]:
                # auto-correlations
                bls = [(i, i) for i in range(nants)]
            else:
                # cross-correlations
                bls = [(i, j) for i in range(nants) for j in range(i+1, nants)]
        else:
            incs = [inc for (bl, inc, pol) in ants]
            if all(incs):
                for (bl, include, pol) in ants:
                    i, j = miriad.bl2ij(bl)
                    bls.append((i, j))
            else:
                bls = [(i, j) for i in range(nants) for j in range(i, nants)] # all baseline pairs
                for (bl, include, pol) in ants:
                    if not include:
                        i, j = miriad.bl2ij(bl)
                        bls.remove((i, j))

        # model visibilities and write data to file
        for (i, j) in bls:
            crd = aa.get_baseline(i,j)
            preamble = (crd, t, (i,j))
            for pol in pols:
                uv['pol'] = a.miriad.str2pol[pol]
                preamble,data,flags = mdl(uv, preamble, None, None)
                if data is None:
                    data = no_data
                    flags = no_flags
                uv.write(preamble, data, flags)


    del(uv)
    
        # bls = [(i, j) for i in range(nants) for j in range(i, nants)] # all baseline pairs
        # for (bl, include, pol) in ants:
        #     if bl == 'auto' and include:
        #         # auto-correlations
        #         bls = [(i, i) for i in range(nants)]
        #     elif bl == 'auto' and not include:
        #         # cross-correlations
        #         bls = [(i, j) for i in range(nants) for j in range(i+1, nants)]
        #     else:
        
        # XXX should really honor opts.ants for which antennas to simulate here.
        # for i,ai in enumerate(aa):
        #     for j,aj in enumerate(aa):
        #         # try:
        #         #     if ai.pol+aj.pol not in pols: continue
        #         #     i,j = ai.num,aj.num
        #         # except(AttributeError): pass
        #         # if j < i: continue
        #         for (bl, include, pol) in ants:
        #             if bl == 'auto' and include:
        #                 if i != j: continue
        #             if
        #         crd = aa.get_baseline(i,j)
        #         preamble = (crd, t, (i,j))
        #         for pol in pols:
        #             uv['pol'] = a.miriad.str2pol[pol]
        #             preamble,data,flags = mdl(uv, preamble, None, None)
        #             if data is None:
        #                 data = no_data
        #                 flags = no_flags
        #             uv.write(preamble, data, flags)
    # del(uv)


