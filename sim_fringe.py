#!/usr/bin/env python

import aipy
import numpy as np
import ephem
import matplotlib.pyplot as plt
import time
import copy
import gc
import os
import numpy.ma as ma
import scipy.interpolate as interpolate


class Antenna_Array(object):
    def __init__(self, antenna_coordinate, antenna_channals, 
                 antenna_diameter, observer_location,tzone="CST", 
                 bandpass_data = None):
        '''
            antenna_coordinate: the coordinate for each antenna in x y z. 
                                in unit of meter
                                x = radial in plane of equator, 
                                y = east, 
                                z = north celestial pole

            antenna_channals: the frequencey channals in unit of GHz
            observer_location: the location (latitude, longitude)
                               in formate of "dgree:minite:second"
        '''
        self.tzone=tzone
        self.get_beam(antenna_channals, antenna_diameter)
        self.ants = []

        # chenge the coordinat to nanoseconds
        light_speed = 3.e8
        antenna_coordinate = np.array(antenna_coordinate)
        self.antenna_coordinate = antenna_coordinate/light_speed * 1.e9

        self.antenna_bandpass = []
        for i in range(antenna_coordinate.shape[0]):
            if bandpass_data != None:
                self.antenna_bandpass.append(self.get_bandpass(antenna_channals, 
                                                               bandpass_data[i]))
            self.ants.append(aipy.phs.Antenna(self.antenna_coordinate[i][0], 
                                              self.antenna_coordinate[i][1],
                                              self.antenna_coordinate[i][2],
                                              self.beam,
                                              delay = 0, offset = 0))

        self.antenna_array = aipy.phs.AntennaArray(ants=self.ants, 
                                                   location=observer_location)

    def get_beam(self, antenna_channals, antenna_diameter, illumination = 0.9):

        self.antenna_channals = antenna_channals
        self.beam = self.creat_beam(self.antenna_channals)
        self.wavelength = 3.e8/(self.antenna_channals*1.e9)
        self.d_ill = np.pi * antenna_diameter * illumination / self.wavelength
        self.beam_pattern = lambda x: (np.sin(self.d_ill*x)/(self.d_ill*x))**2

    def get_source_list(self, source_name_list=[], source_coordinate_list=[]):

        for name in source_name_list:
            self.srcs.append(aipy.phs.RadioSpecial(name))
        for coor in source_coordinate_list:
            self.srcs.append(aipy.phs.RadioFixedBody(coor[0], coor[1], 100, coor[2]))
            source_name_list.append(coor[2])

        return source_name_list


    def get_juliantime_steps(self, start, duration, interval):
        '''
            start time: in form of "year/month/day hour:minute:second"
            duration  : in unit of second
            interval  : in unit of second
        '''
        start_julian = ephem.julian_date(start)
        duration /= (24.*60.*60) # change to unit of day
        interval /= (24.*60.*60) # change to unit of day
    
        time_steps = np.arange(0, duration, interval)
    
        time_steps += start_julian
    
        return time_steps


    def creat_beam(self, freqs_channal):
        #freqs_channal = np.linspace(.700, .800, 1024)
        return aipy.phs.Beam(freqs_channal)

    def get_bandpass(self, freqs_channal, bandpass_file, unit=1.e-3, plot=False):
        '''
            unit: bandpass_data * unit -> GHz
        '''

        bandpass_data = np.load(bandpass_file)
        selected = bandpass_data[1,...] != 0
        bandpass_f = interpolate.interp1d(bandpass_data[0,...][selected]*unit, 
                                          bandpass_data[1,...][selected])

        bandpass = bandpass_f(freqs_channal)
        if plot:
            plt.figure(figsize=(8,8))
            plt.plot(bandpass_data[0]*unit, bandpass_data[1], label='raw data')
            plt.plot(freqs_channal, bandpass, label='interpolated')
            plt.xlim(xmin=freqs_channal.min(), xmax=freqs_channal.max())
            plt.legend()
            plt.show()
            plt.close()

        return bandpass

    def get_gain(self, pointing, source_az, source_alt):
        ''' pointing is antenna direction by azimuth and altitude
        '''
        def convert_dms_2_d(dms):
            dms = dms.split(':')
            return float(dms[0]) + (float(dms[1]) + float(dms[2])/60.)/60.

        pointing_az = convert_dms_2_d(pointing[0])*np.pi/180.
        pointing_alt= convert_dms_2_d(pointing[1])*np.pi/180.

        source_az   = convert_dms_2_d(source_az)*np.pi/180.
        source_alt  = convert_dms_2_d(source_alt)*np.pi/180.

        delta_y = source_alt - pointing_alt
        delta_az= source_az - pointing_az
        delta_x = 2.*np.arcsin(np.cos(source_alt)*np.sin(0.5*delta_az))

        delta = np.sqrt( delta_x**2 + delta_y**2 )

        return self.beam_pattern(delta)

    def ensure_dir(self, f):
        '''create dir if dir not exit
        '''
        #d = os.path.dirname(f)
        if not os.path.exists(f):
            os.makedirs(f)
    

    def observation(self, start_time, duration, interval_time,
                    source_name_list = [], source_coordinate_list=[], 
                    pointing = None):
        '''
            start_time: the start time for observation
                        in form of "year/month/day hour:minute:second"
            duration: time used in unit of second
            interval_time: the interval time between time steps in unit of second
        '''

        self.srcs = []
        self.srcs_name = self.get_source_list(source_name_list, 
                                              source_coordinate_list)
        self.cat = aipy.phs.SrcCatalog(self.srcs)

        start_time_gmt = time.strftime("%Y/%m/%d %H:%M:%S",
                         time.gmtime(time.mktime(time.strptime(
                         start_time+" "+self.tzone, "%Y/%m/%d %H:%M:%S %Z"))))  
                         # if there is not tzone, gmtime will read tzone from system 
                         # enviroment variable
        time_steps = self.get_juliantime_steps(start_time_gmt, duration, interval_time)

        source_phs = {}
        source_amp = {}
        source_img = {}
        source_rel = {}
        for source_name in self.srcs_name:
            source_phs[source_name] = {}
            source_amp[source_name] = {}
            source_img[source_name] = {}
            source_rel[source_name] = {}

            for i in range(len(self.ants)):
                for j in range(i+1, len(self.ants)):
                    source_phs[source_name]['%d%d'%(i,j)] = []
                    source_amp[source_name]['%d%d'%(i,j)] = []
                    source_img[source_name]['%d%d'%(i,j)] = []
                    source_rel[source_name]['%d%d'%(i,j)] = []

            for time_step in time_steps:
            
                self.antenna_array.set_jultime(time_step)
                
                self.cat.compute(self.antenna_array)

                if pointing != None:
                    gain = self.get_gain(pointing, str(self.cat[source_name].az), 
                                                   str(self.cat[source_name].alt))
                else:
                    gain = 1.

                base_line_num = 0
                for i in range(len(self.ants)):
                    for j in range(i+1, len(self.ants)):
                        data = np.array(self.antenna_array.gen_phs(
                                        src=self.cat[source_name], 
                                        i=i, j=j)).conj()
                        if len(self.antenna_bandpass) != 0:
                            bandpass_gain = np.sqrt(self.antenna_bandpass[i]*
                                                    self.antenna_bandpass[j])
                        else:
                            bandpass_gain = 1.


                        data *= bandpass_gain
                        data *= gain
            
                        phase_angle = np.angle(data)

                        ## add frequency depended shift
                        #phase_shift = np.linspace(0., np.pi, 
                        #              len(self.antenna_channals))
                        #phase_angle += phase_shift


                        # switch phase angle to range [0, 2*pi]
                        #phase_angle = phase_angle%(2*np.pi)

                        phase_selec = copy.deepcopy(gain)
                        phase_selec[gain>0.1]  = 1.
                        phase_selec[gain<=0.1] = 0.
                        phase_angle *= phase_selec

                        amp = np.abs(data)

                        base_line_num += 1
            
                        source_phs[source_name]['%d%d'%(i,j)].append(phase_angle)
                        source_amp[source_name]['%d%d'%(i,j)].append(amp)
                        source_rel[source_name]['%d%d'%(i,j)].append(data.real)
                        source_img[source_name]['%d%d'%(i,j)].append(data.imag)

            ###########
            figsize=(22, 6)
            file_name = time.strftime("%Y_%m_%d_%H_%M_%S", 
                       time.strptime(start_time, "%Y/%m/%d %H:%M:%S"))
            dir_root = './'
            dir_file= dir_root + file_name
            self.ensure_dir(dir_file)
            ##########
            print "==== begin plot  2d ===="

            base_lines = source_phs[source_name].keys()
            for i in range(base_line_num):
                plt.figure(figsize=figsize)
                base_line = base_lines[i]
                x_axis = np.linspace(0, duration, len(time_steps))
                y_axis = self.antenna_channals
                X, Y = np.meshgrid(x_axis, y_axis)
                plt.pcolor(X, Y, np.array(source_phs[source_name][base_line]).T)
                plt.xlim(xmin=x_axis.min(), xmax=x_axis.max())
                plt.xlabel('time [s]')
                plt.ylim(ymin=y_axis.min(), ymax=y_axis.max())
                plt.ylabel('frequency [GHz] baseline %s'%base_line)
                plt.colorbar()
                plt.savefig(dir_file + '/phs_%s_%s_ch%s.png'%(
                            source_name, file_name, base_line ))
                np.save(dir_file+'/phs_%s_%s_ch%s.npy'%(
                            source_name, file_name, base_line ), 
                            np.array(source_phs[source_name][base_line]).T)
                plt.close()
                gc.collect()
            base_lines = source_amp[source_name].keys()
            for i in range(base_line_num):
                plt.figure(figsize=figsize)
                base_line = base_lines[i]
                x_axis = np.linspace(0, duration, len(time_steps))
                y_axis = self.antenna_channals
                X, Y = np.meshgrid(x_axis, y_axis)
                plt.pcolor(X, Y, np.array(source_amp[source_name][base_line]).T)
                plt.xlim(xmin=x_axis.min(), xmax=x_axis.max())
                plt.xlabel('time [s]')
                plt.ylim(ymin=y_axis.min(), ymax=y_axis.max())
                plt.ylabel('frequency [GHz] baseline %s'%base_line)
                plt.colorbar()
                plt.savefig(dir_file+'/amp_%s_%s_ch%s.png'%(
                            source_name, file_name, base_line))
                np.save(dir_file+'/amp_%s_%s_ch%s.npy'%(
                            source_name, file_name, base_line ), 
                            np.array(source_amp[source_name][base_line]).T)
                plt.close()
                gc.collect()
            base_lines = source_rel[source_name].keys()
            for i in range(base_line_num):
                plt.figure(figsize=figsize)
                base_line = base_lines[i]
                x_axis = np.linspace(0, duration, len(time_steps))
                y_axis = self.antenna_channals
                X, Y = np.meshgrid(x_axis, y_axis)
                plt.pcolor(X, Y, np.array(source_rel[source_name][base_line]).T)
                plt.xlim(xmin=x_axis.min(), xmax=x_axis.max())
                plt.xlabel('time [s]')
                plt.ylim(ymin=y_axis.min(), ymax=y_axis.max())
                plt.ylabel('frequency [GHz] baseline %s'%base_line)
                plt.colorbar()
                plt.savefig(dir_file+'/rel_%s_%s_ch%s.png'%(
                            source_name, file_name, base_line))
                np.save(dir_file+'/rel_%s_%s_ch%s.npy'%(
                            source_name, file_name, base_line ), 
                            np.array(source_rel[source_name][base_line]).T)
                plt.close()
                gc.collect()
            base_lines = source_img[source_name].keys()
            for i in range(base_line_num):
                plt.figure(figsize=figsize)
                base_line = base_lines[i]
                x_axis = np.linspace(0, duration, len(time_steps))
                y_axis = self.antenna_channals
                X, Y = np.meshgrid(x_axis, y_axis)
                plt.pcolor(X, Y, np.array(source_img[source_name][base_line]).T)
                plt.xlim(xmin=x_axis.min(), xmax=x_axis.max())
                plt.xlabel('time [s]')
                plt.ylim(ymin=y_axis.min(), ymax=y_axis.max())
                plt.ylabel('frequency [GHz] baseline %s'%base_line)
                plt.colorbar()
                plt.savefig(dir_file+'/img_%s_%s_ch%s.png'%(
                            source_name, file_name, base_line))
                np.save(dir_file+'/img_%s_%s_ch%s.npy'%(
                            source_name, file_name, base_line ), 
                            np.array(source_rel[source_name][base_line]).T)
                plt.close()
                gc.collect()

            print "==== begin plot  1d ===="
            base_lines = source_rel[source_name].keys()
            for i in range(base_line_num):
                plt.figure(figsize=figsize)
                base_line = base_lines[i]
                x_axis = np.linspace(0, duration, len(time_steps))
                #one_D=ma.mean(np.array(source_rel[source_name][base_line]), axis=1)
                one_D=np.array(source_rel[source_name][base_line])[:,150]
                plt.plot(x_axis, one_D)
                plt.title("real part, integration on Frequence band for channel "
                          + base_line)
                plt.xlabel('time[s]')
                plt.ylabel('amplitude')
                plt.xlim(xmin=x_axis.min(), xmax=x_axis.max())
                plt.savefig(dir_file+'/rel_%s_%s_ch%s_t_1d.png'%(
                            source_name, file_name, base_line))
                np.save(dir_file+'/rel_%s_%s_ch%s_t_1d.npy'%(
                            source_name, file_name, base_line), 
                            np.concatenate( [x_axis, one_D], axis=0))
                plt.close()
                gc.collect()


if __name__ == "__main__":

#    aa = Antenna_Array([[0, 28.87, 0], [0, 0, 0], [0, -14.02, 0]], 
#                       np.linspace(0.7, 0.8, 128), 5.,
#                       #np.linspace(0.135, 0.250, 128), 
#                       ("42:12:57.24","115:14:45.96"))
#    #aa = Antenna_Array([[0, -14.02, 0], [0, 0, 0], [0, 28.87, 0]], 
#    #                   np.linspace(0.7, 0.8, 128), 
#    #                   ("42:12:57.24","115:14:45.96"))
#
#    #aa.observation("2013/7/13 8:30:00", 3282, 10.,
#    #               source_name_list = ['Sun',], 
#    #               source_coordinate_list=[])
#
#    #aa.observation("2013/7/12 5:00:00", 14*60*60., 20.,
#    #               source_name_list = ['Sun',], 
#    #               source_coordinate_list=[],)
#
#    aa.observation("2013/7/12 11:00:00", 2*60*60., 2.,
#                   source_name_list = ['Sun',], 
#                   source_coordinate_list=[],
#                   pointing=["163.82:0:0",  "69.37:0:0"])

    #aa.observation("2013/7/12 18:25:16", 3282., 10.,
    #               source_name_list = ['Sun',], 
    #               source_coordinate_list=[])

    #aa.observation("2013/7/12 20:48:32", 25.*60., 10.,
    #               source_name_list = [], 
    #               source_coordinate_list=[['19:59:28.36','40:44:2.10','Cygnus_A'],])

    #aa.observation("2013/7/12 22:49:47", 25.*60., 10.,
    #               source_name_list = [], 
    #               source_coordinate_list=[['23:23:26.00','58:48:0.0','Cassiopeia_A'],])

    #aa = Antenna_Array([[0, 0, 0],[0, 30, 0]],
    #                   np.linspace(1.200, 1.250, 128), 
    #                   ("40:23:53.95", "117:35:13.03"))

    #aa.observation("2013/1/3 18:20:38", 7380., 10.,
    #               source_name_list = [], 
    #               source_coordinate_list=[['23:23:26.00','58:48:0.0','Cassiopeia_A'],])

##########################---inner mongolia 2013 9 18--------------------
    bandpass_data = ['./data/9_17_15_31_1.dat-11-real-f-1d.npy',
                     './data/9_17_15_31_1.dat-11-real-f-1d.npy',
                     './data/9_17_15_31_1.dat-55-real-f-1d.npy']
    aa = Antenna_Array([[0, 28.87, 0], [0, 0, 0], [0, -14.02, 0]], 
                       np.linspace(0.7, 0.8, 256), 5.,
                       ("42:12:57.24","115:14:45.96"),
                       bandpass_data = bandpass_data)

#    aa = Antenna_Array([[0, 29, 0], [0, 0, 0], [0, -14.02, 0]], 
#                       np.linspace(0.7, 0.8, 256), 5.,
#                       ("42:12:57.24","115:14:45.96"))

    pointing = ["248:16:49",  "25:8:39"] # we observe sun  9-17-16-02-41
    pointing = ["248:16:22",  "25:9:3" ] # we observe sun  9-17-16-02-41  ##yichao

    aa.observation("2013/9/17 15:31:1", 4000., 5.,
                   source_name_list = ['Sun',], 
                   source_coordinate_list=[], 
                   pointing=pointing)
                   



#    aa.observation("2013/7/12 11:00:00", 2*60*60., 20.,
#                   source_name_list = ['Sun',], 
#                   source_coordinate_list=[],
#                   pointing=["163.82:0:0",  "69.37:0:0"])
