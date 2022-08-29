#!/usr/env python3
# -*- encoding: utf-8 -*-

# ======================================================================
# LSST-VRO / StarDICE
#
# Low level control for the Starlight Express Oculus All-Sky Camera
# Python3 minimal driver based on indilib 3rd party drivers available at :
# https://github.com/indilib/indi-3rdparty/tree/master/indi-sx
# ======================================================================

# ======================================================================
# The All-Sky Camera has a 180-degrees FOV, coupled with an uncooled CCD camera
# at resolution 1392x1040.
# ======================================================================

# ======================================================================
# Authors: K. Sommer, L. Le Guillou, B. Plez 
# Email: <kelian.sommer@umontpellier.fr>, <llg@lpnhe.in2p3.fr>, <bertrand.plez@umontpellier.fr>
# ======================================================================

import sys
import usb.core
import numpy as np
import struct
import matplotlib.pyplot as plt
import time
from datetime import datetime
from astropy.io import fits

class AllSkyCamera(object):
    """
    Class to control the Starlight Express Oculus AllSky Camera :
    https://www.sxccd.com/product/oculus-all-sky-camera-180/
    Interface implemented based on INDI driver :
    https://github.com/indilib/indi-3rdparty/tree/master/indi-sx
    """

    # Communication parameters
    PRODUCT_ID = 0x509
    VENDOR_ID = 0x1278
    USB_REQ_DATAOUT = 0x00
    USB_REQ_DATAIN = 0x80
    USB_REQ_VENDOR = (2 << 5)
    BULK_IN = 0x0082
    BULK_OUT = 0x0001
    BULK_COMMAND_TIMEOUT = 2000
    BULK_DATA_TIMEOUT = 40000 #Older SXV-M25C takes 14s unbinned
    CHUNK_SIZE = 4 * 1024 * 1024

    # CCD camera control commands / properties.
    SXUSB_GET_FIRMWARE_VERSION = 255
    SXUSB_ECHO = 0
    SXUSB_READ_PIXELS_DELAYED = 2
    SXUSB_RESET = 6
    SXUSB_GET_CCD = 8
    SXUSB_READ_PIXELS_GATED = 18

    # Image properties
    CAMERA = 'SX Oculus'    
    WIDTH = 1392
    HEIGHT = 1040
    IMG_SIZE = 2*WIDTH*HEIGHT
    PXSIZE = 4.6484375
    IMTYPE = 'All Sky'

    def __init__(self, vendor_id  = VENDOR_ID, product_id = PRODUCT_ID, debug = True):
        """Create a StarLight Express Oculus instance

        Parameters
        ----------
        vendor_id : bytes
            Id of the vendor
        product_id : bytes
            Id of the product
        debug : bool
            If True, print additional information (get and set commands in raw format)
        """
        
        self.vendor_id = vendor_id
        self.product_id = product_id
        self.device = None
        self.ifaces = []
        self.iface_id = None
        self.debug = debug

    def open(self):
        """
        Detect the camera on the USB bus. May fail.
        Open and initialize the USB device to communicate with the camera.
        """
        
        # ---- look for the USB device --------

        if self.debug: 
            print("Looking for USB device %04x:%04x ..." %
                  (self.vendor_id, self.product_id),
                  file=sys.stderr)
        
        self.device = usb.core.find(idVendor=self.vendor_id, idProduct=self.product_id)

        if not(self.device):
            raise IOError("Cannot find USB device %04x:%04x. Stop." %
                          (self.vendor_id, self.product_id))

        if self.debug: 
            print("USB device %04x:%04x found." %
                  (self.vendor_id, self.product_id),
                  file=sys.stderr)
        
        # ---- finding config, interfaces, endpoints

        self.cfg = self.device[0] # Only one config
        self.ifaces = self.cfg.interfaces()
        self.iface = self.ifaces[0]
        self.iface_id = self.iface.bInterfaceNumber

        if self.debug: 
            print("Testing communication with USB device %04x:%04x ..." %
                  (self.vendor_id, self.product_id),
                  file=sys.stderr)

        # Reset the USB device
        self.device.reset()

        # If a kernel module is managing the device,
        # ask the kernel to release the handle
        #
        if self.device.is_kernel_driver_active(self.iface_id):
            dev.detach_kernel_driver(iface_id)

        # ---- Now is the time for an Echo test 
        # Sending "INFO?" request
        # and checking the answer

        if not(self.get_firmware_version()):
            raise IOError("No echo from USB device  %04x:%04x ..." %
                          (self.vendor_id, self.product_id))

        if self.debug: 
            print("Testing communication with USB device %04x:%04x successful." %
                  (self.vendor_id, self.product_id),
                  file=sys.stderr)        

    def close(self):
        """Close the USB device."""
        
        if self.device:
            ## TODO: find a proper way to release the USB device
            del self.device

        self.device = None

    def __del__(self):
        self.close()

    def get_camera_params(self):
        """Print all camera parameters"""
        
        cmd = bytearray([self.USB_REQ_VENDOR | self.USB_REQ_DATAIN, self.SXUSB_GET_CCD, 0, 0, 0, 0, 17, 0])
        if self.debug:
            print('ALL-SKY CAMERA : SENDING : ', cmd)
        self.device.write(self.BULK_OUT, cmd, self.BULK_COMMAND_TIMEOUT)
        
        res = self.device.read(self.BULK_IN, 17, self.BULK_DATA_TIMEOUT)

        width            = res[2] | (res[3] << 8)
        height           = res[6] | (res[7] << 8)
        pix_width        = ((res[8] | (res[9] << 8)) / 256.0)
        pix_height       = ((res[10] | (res[11] << 8)) / 256.0)
        bits_per_pixel   = res[14]

        print('WIDTH : {}'.format(width))
        print('HEIGHT : {}'.format(height))
        print('PIX WIDTH : {}'.format(pix_width))
        print('PIX HEIGHT : {}'.format(pix_height))
        print('BIT DEPTH : {}'.format(bits_per_pixel))

    def get_firmware_version(self):
        """Get firmware version of the camera. Used as an "echo test" procedure.

        Returns
        -------
        firmware_version : int
            Actual firmware version, should be 65553
        """
        cmd = bytearray([self.USB_REQ_VENDOR | self.USB_REQ_DATAIN, self.SXUSB_GET_FIRMWARE_VERSION, 0, 0, 0, 0, 4, 0])
        if self.debug:
            print('ALL-SKY CAMERA : SENDING : ', cmd)
        self.device.write(self.BULK_OUT, cmd, self.BULK_COMMAND_TIMEOUT)
        res = self.device.read(self.BULK_IN, 4, self.BULK_DATA_TIMEOUT)
        firmware_version = (res[0] | (res[1] << 8) | (res[2] << 16) | (res[3] << 24))

        if firmware_version == 65563:
            print('ALL-SKY CAMERA : CONNECTED, FIRMWARE VERSION = {}'.format(firmware_version))
        else:
            print('ALL-SKY CAMERA : CANNOT CONNECT, {}'.format(firmware_version))
        return firmware_version


    def take_exposure(self, exp_time_ms = 1000):
        """Take a single frame exposure
        
        Parameters
        ----------
        exp_time_ms : int
            Exposure time in milliseconds
        """
        cmd = [self.USB_REQ_VENDOR | self.USB_REQ_DATAOUT, self.SXUSB_READ_PIXELS_DELAYED, 0, 0, 0, 0, 14, 0, 0, 0, 0, 0, self.WIDTH & 0xFF, self.WIDTH >> 8, self.HEIGHT & 0xFF, self.HEIGHT >> 8, 1, 1, exp_time_ms & 0xFF, (exp_time_ms >> 8) & 0xFF, (exp_time_ms >> 16) & 0xFF, (exp_time_ms >> 24) & 0xFF]
        if self.debug:
            print('ALL-SKY CAMERA : SENDING : ', cmd)
        self.device.write(self.BULK_OUT, cmd)
        print('ALL-SKY CAMERA : TAKE {} MS EXPOSURE'.format(exp_time_ms))
        self.timestamp = datetime.now()

    def read_image(self, display = False):
        """Read a single image from the camera's buffer
        
        Parameters
        ----------
        display : bool
            If True, plot the image

        Returns
        -------
        img : numpy.array
            2D array of shape (self.HEIGHT, self.WIDTH) containing raw image data in 16-bits
        """

        img_buff = self.device.read(self.BULK_IN, self.IMG_SIZE, self.BULK_DATA_TIMEOUT)
        img_strip = struct.unpack('@%dH'%(self.IMG_SIZE/2), img_buff)
        img = np.array(img_strip).reshape(self.HEIGHT, self.WIDTH)

        if display == True:
            plt.imshow(img)
            plt.show()

        return img

    def write_to_fits(self, img, path):
        """Write data to FITS file
        
        Parameters
        ----------
        img : numpy.array
            2D array of shape (self.HEIGHT, self.WIDTH) containing raw image data in 16-bits
        path : str
            Directory to write the FITS file
        """
        hdu = fits.PrimaryHDU(img)
        hdu.scale('uint16')
        hdu.header['CAMERA'] = self.CAMERA
        hdu.header['PXSIZE'] = self.PXSIZE
        hdu.header['IMTYPE'] = self.IMTYPE
        hdu.header['DATE-OBS'] = self.timestamp.strftime('%Y-%m-%dT%H:%M:%S')
        
        try:
            hdu.writeto(path+'/'+self.timestamp.strftime('%Y_%m_%d_%H_%M_%S_%f_allsky.fits'))
            if self.debug:
                print('{} : ALL-SKY CAMERA : IMAGE WRITTEN TO FITS FILE'.format(self.timestamp.strftime('%Y-%m-%dT%H:%M:%S')))
        except:
            print('{} : ALL-SKY CAMERA : COULD NOT WRITE IMAGE TO FITS FILE'.format(self.timestamp.strftime('%Y-%m-%dT%H:%M:%S')))
        
    def reset(self, wait_for_reset = 10):
        """Reset the camera
        
        Parameters
        ----------
        wait_for_reset : int
            Delay in seconds to let the camera reset before sending new command
        """
        cmd = bytearray([self.USB_REQ_VENDOR | self.USB_REQ_DATAOUT, self.SXUSB_RESET, 0, 0, 0, 0, 0, 0])
        if self.debug:
            print('ALL-SKY CAMERA : SENDING : ', cmd)
        self.device.write(self.BULK_OUT, cmd, self.BULK_COMMAND_TIMEOUT)
        time.sleep(wait_for_reset)
