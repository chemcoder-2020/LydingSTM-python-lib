class PyTai:
    """Create a class for the Spectra-Physics MaiTai laser"""

    def __init__(self,port,baudrate=38400,bits=8,parity='N',stop_bits=1,timeout=0,xonxoff=1):
        import serial
        self.ser = serial.Serial(port, baudrate, bits, parity, stop_bits, timeout, xonxoff)
        self.port=self.ser.port
        self.baudrate=self.ser.baudrate
        self.bits=self.ser.bytesize
        self.parity=self.ser.parity
        self.stop_bits=self.ser.stopbits
        self.timeout=self.ser.timeout
        self.xonxoff=self.ser.xonxoff

    def query(self,command):
        __instr_response=''
        self.ser.open()
        self.ser.write("{c}\n".format(c=command).encode('utf-8'))
        __instr_response=self.ser.readall()
        self.ser.close()
        return __instr_response

    #  Write-only commands
    def off(self):
        self.ser.open()
        self.ser.write('SAVE\n'.encode('utf-8'))
        self.ser.write('OFF\n'.encode('utf-8'))
        self.ser.close()

    def on(self):
        self.ser.open()
        self.ser.write('ON\n'.encode('utf-8'))
        self.ser.close()

    def save_status(self):
        self.ser.open()
        self.ser.write('SAVE\n'.encode('utf-8'))
        self.ser.close()

    def set_wavelength(self,wav):
        import time
        self.ser.open()
        self.ser.write('WAVELENGTH {l}\n'.format(l=wav).encode('utf-8'))
        self.ser.close()
        recentWav = self.recent_wavelength_commanded()
        while True:
            curWav = self.MaiTai_wavelength()
            print(curWav.decode("ascii"))
            if curWav == recentWav:
                print(curWav.decode("ascii"), 'Done')
                break
            time.sleep(3)
        print('Yup, Done!!')

    def shutter_off(self):
        self.ser.open()
        self.ser.write('SHUTTER 0\n'.encode('utf-8'))
        self.ser.close()

    def shutter_on(self):
        self.ser.open()
        self.ser.write('SHUTTER 1\n'.encode('utf-8'))
        self.ser.close()

    #  Query commands
    def recent_wavelength_commanded(self):
        return self.query('WAVelength?')

    def wavelength_range(self):
        maxWav = self.query('WAVelength:MAX?')
        minWav = self.query('WAVelength:MIN?')
        return minWav, maxWav

    def idYourself(self):
        return self.query('*IDN?')

    def shutter_status(self):
        import time
        time.sleep(2)
        shutStat = self.query('SHUTter?')
        if shutStat.decode("ascii") == "0":
            print('Shutter Closed')
        else:
            print('Shutter Open')
        return shutStat

        ## The pump laser: Millenia
    def pump_laser_which_mode(self):
        return self.query('MODE?')

    def pump_laser_history(self):
        return self.query('PLASER:AHISTORY?')

    def pump_laser_errcode(self):
        return self.query('PLASER:ERRCODE?')

    def pump_laser_current_commanded(self):
        return self.query('PLASER:CURRENT?')

    def pump_laser_power_commanded(self):
        return self.query('PLASER:POWER?')

    def pump_laser_diodes_currents(self):
        '''Typical response may be “75.1%<LF>”'''
        __d1Curr = self.query('READ:PLASer:DIODe1:CURRent?')
        __d2Curr = self.query('READ:PLASer:DIODe2:CURRent?')
        return __d1Curr, __d2Curr

    def pump_laser_diodes_temps(self):
        '''Typical response may be “20.5<LF>”'''
        __d1Temp = self.query('READ:PLASer:DIODe1:TEMPerature?')
        __d2Temp = self.query('READ:PLASer:DIODe2:TEMPerature?')
        return __d1Temp, __d2Temp

    def pump_laser_current_actual(self):
        return self.query('READ:PLASer:PCURrent?')

    def pump_laser_power_actual(self):
        return self.query('READ:PLASer:POWer?')

    def pump_laser_SHG(self):
        return self.query('READ:PLASer:SHGS?')

        ## The MaiTai laser
    def MaiTai_is_modelocked(self):
        return self.query('CONTROL:MLENABLE?')

    def MaiTai_history(self):
        return self.query('READ:AHISTORY?')

    def is_warmed_up(self):
        '''response is in the form of b'x%'. Laser can be turned on when the response is b'100%' '''
        return self.query('READ:PCTWARMEDUP?')

    def MaiTai_output_power(self):
        return self.query('READ:POWer?')

    def MaiTai_wavelength(self):
        return self.query('READ:WAVelength?')