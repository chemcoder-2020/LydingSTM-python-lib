import time
import serial
import numpy as np
import struct

class Motor:
    def __init__(self, name, ser):
        self.motor = name
        self.ser = ser
        self.angle_resolution = 360 / 400 / 32
        self.codeDict = {0:"MOTOR A SPECS: Sparkfun Bipolar Stepper Motor: 0.9 deg/step, 1.7A/phase. Motor controller: DRV8834, 1/32 microstepping.",
                        1:"MOTOR B SPECS: Sparkfun Bipolar Stepper Motor: 0.9 deg/step, 1.7A/phase. Motor controller: DRV8834, 1/32 microstepping."}

    def query(self, command,wait=1.1):
        # if self.ser.is_open == False:
        #     self.ser.open()
        self.ser.reset_input_buffer()
        self.write(command)
        time.sleep(wait)
        # _instr_response = self.ser.readall()
        _instr_response = self.ser.readline().split(b"\r\n")[0]

        # return _instr_response.decode('utf-8')
        return _instr_response

    def write(self, command):
        self.ser.write("{n}{c}\n".format(n=self.motor, c=command.strip("\n")).encode('utf-8'))

    def current_angle(self):
        # if self.ser.is_open == False:
        #     self.ser.open()
        cmd = "curr_angle\n"
        result = self.query(cmd)

        return struct.unpack("<f",result)[0]

    def IDN(self):
        # if self.ser.is_open == False:
        #     self.ser.open()
        cmd = '*IDN?'
        result = self.query(cmd)
        code = struct.unpack("<b",result)[0]
        print(self.codeDict[code])
    # IDN and current_angle both returns a value to Python, so query was needed

    def sethome(self):
        # if self.ser.is_open == False:
        #     self.ser.open()
        cmd = 'sethome\n'
        self.write(cmd)
        print("The home for motor %s is set." % self.motor)

    def home(self):
        # if self.ser.is_open == False:
        #     self.ser.open()
        cmd = 'home\n'
        self.write(cmd)
        while True:
            time.sleep(1.1)
            stat = self.was_homed()
            if stat:
                break
        print("Motor {a} is homed. Current angle location is 0.000 degrees".format(a=self.motor))
        # self.ser.close()

    def was_homed(self):
        cmd = 'home_status\n'
        result = self.query(cmd)
        try:
            return bool(int(struct.unpack("<b",result)[0]))
        except:
            return False

    def set_home_status(self, status):
        cmd = 'set_home_status_to_' + str(status).lower()
        self.write(cmd)
        print("Homed status set to {a}".format(a=status))

    def sleep(self):
        # if self.ser.is_open == False:
        #     self.ser.open()
        cmd = 'sleep\n'
        self.write(cmd)
        # self.ser.close()
        time.sleep(2)
        self.set_home_status(False)
        print("Motor %s is asleep. Homed status set to False" % self.motor)

    def wake(self):
        # if self.ser.is_open == False:
        #     self.ser.open()
        cmd = 'wake\n'
        self.write(cmd)
        # self.ser.close()
        print("Motor %s is woke." % self.motor)

    def step_low(self):
        cmd = 'step_low\n'
        self.write(cmd)

    def step_high(self):
        cmd = 'step_high\n'
        self.write(cmd)

    def jog_low(self):
        cmd = 'jog_low\n'
        self.write(cmd)

    def jog_high(self):
        cmd = 'jog_high\n'
        self.write(cmd)

    def rotate_motor(self, degrees):
        # if self.ser.is_open == False:
        #     self.ser.open()
        # modulo = degrees % self.angle_resolution if degrees >= 0 else -(np.abs(degrees) % self.angle_resolution)
        # if np.isclose(modulo,0):
        #     mod_degrees = np.round(degrees,2)
        # else:
        #     mod_degrees = np.round(degrees,2) - modulo + self.angle_resolution if modulo/2 > self.angle_resolution \
        #                         else np.round(degrees,2) - modulo
        steps = np.int(np.round(degrees / self.angle_resolution))
        mod_degrees = np.round(steps * self.angle_resolution,3)
        if -180 <= mod_degrees <= 180:
            cmd = str(mod_degrees) + "\n"
            self.write(cmd)
            print('Motor {a} current angle location: {b:.3f} degrees'.format(a=self.motor, b=np.round(mod_degrees,3)))
        else:
            print('Degrees must be between 180 and -180.')


class PyMotor:
    def __init__(self, port, baudrate = 38400, bits = 8, parity = 'N', stop_bits = 1, timeout = 0, xonxoff = False):
        self.ser = serial.Serial(port, baudrate, bits, parity, stop_bits, timeout, xonxoff)
        self.port = self.ser.port
        self.baudrate = self.ser.baudrate
        self.bits = self.ser.bytesize
        self.parity = self.ser.parity
        self.stop_bits = self.ser.stopbits
        self.timeout = self.ser.timeout
        self.xonxoff = self.ser.xonxoff
        self.motorA = Motor("A", self.ser)
        self.motorB = Motor("B", self.ser)
# serial is a class and an instance of serial (self.ser) was created to specify the port

    def close(self):
        self.ser.close()
        print("Device status: {a}".format(a="Open" if self.ser.is_open else "Closed"))

    def open(self):
        self.ser.open()
        print("Device status: {a}".format(a="Open" if self.ser.is_open else "Closed"))

    def query(self, command):
        # if self.ser.is_open == False:
        #     self.ser.open()
        self.ser.reset_input_buffer()
        self.write(command)
        time.sleep(3)
        # _instr_response = self.ser.readall()
        _instr_response = self.ser.readline().split(b"\r\n")[0]

        # return _instr_response.decode('utf-8')
        return _instr_response
# query was made to make sure certain functions write and also read and return something to Python

    def write(self, command):
        self.ser.write("{c}\n".format(c = command.strip("\n")).encode('utf-8'))

    def readall(self):
        self.ser.readall()
