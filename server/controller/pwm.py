#! /usr/bin/env python3

"""PCA9685 16-Channel PWM module driver
when run as main, Vout on channel0 set to about 1.66, Vout on channel1 set to
about 0.83
"""
import math
import platform
import sys
import time


class PCA9685():
    """PWM module driver for Jetson AGX Xavier
    """
    # Registers/etc.
    __SUBADR1 = 0x02
    __SUBADR2 = 0x03
    __SUBADR3 = 0x04
    __MODE1 = 0x00
    __MODE2 = 0x01
    __PRESCALE = 0xFE
    __LED0_ON_L = 0x06
    __LED0_ON_H = 0x07
    __LED0_OFF_L = 0x08
    __LED0_OFF_H = 0x09
    __ALLLED_ON_L = 0xFA
    __ALLLED_ON_H = 0xFB
    __ALLLED_OFF_L = 0xFC
    __ALLLED_OFF_H = 0xFD

    def __init__(self, address=0x40, bus=8, debug=False):
        import smbus
        self.bus = smbus.SMBus(bus)
        self.address = address
        self.channels_number = 16  # number of channels under control
        self.debug = debug
        if (self.debug):
            print("Reseting PCA9685")
        self.write(self.__MODE1, 0x00)
        self.setPWMFreq()

    def write(self, reg, value):
        "Writes an 8-bit value to the specified register/address"
        self.bus.write_byte_data(self.address, reg, value)
        if (self.debug):
            print("I2C: Write 0x%02X to register 0x%02X" % (value, reg))

    def read(self, reg):
        "Read an unsigned byte from the I2C device"
        result = self.bus.read_byte_data(self.address, reg)
        if (self.debug):
            print("I2C: Device 0x%02X returned 0x%02X from reg 0x%02X" % (self.address, result & 0xFF, reg))
        return result

    def setPWMFreq(self, freq=50):
        "Sets the PWM frequency"
        prescaleval = 25000000.0    # 25MHz
        prescaleval /= 4096.0       # 12-bit
        prescaleval /= float(freq)
        prescaleval -= 1.0
        if (self.debug):
            print("Setting PWM frequency to %d Hz" % freq)
            print("Estimated pre-scale: %d" % prescaleval)
        prescale = math.floor(prescaleval + 0.5)
        if (self.debug):
            print("Final pre-scale: %d" % prescale)
        oldmode = self.read(self.__MODE1)
        newmode = (oldmode & 0x7F) | 0x10        # sleep
        self.write(self.__MODE1, newmode)        # go to sleep
        self.write(self.__PRESCALE, int(math.floor(prescale)))
        self.write(self.__MODE1, oldmode)
        time.sleep(0.005)
        self.write(self.__MODE1, oldmode | 0x80)
        self.write(self.__MODE2, 0x04)

    def setPWM(self, channel, on, off):
        "Sets a single PWM channel"
        self.write(self.__LED0_ON_L + 4 * channel, on & 0xFF)
        self.write(self.__LED0_ON_H + 4 * channel, on >> 8)
        self.write(self.__LED0_OFF_L + 4 * channel, off & 0xFF)
        self.write(self.__LED0_OFF_H + 4 * channel, off >> 8)
        if (self.debug):
            print("channel: %d  LED_ON: %d LED_OFF: %d" % (channel, on, off))

    def setValue(self, channel, percentage):
        """set percentage of pwm duty cycle

        Parameters
        ----------
        channel : int
            channel 0-15
        percentage : float
            0-1
        """
        self.setPWM(channel, 0, int(4095 * percentage))

    def reset_all(self):
        """reset duty of all 16 channels to 0
        """
        for c in range(self.channels_number):
            self.setValue(c, 0)
        print('⚙️  PWM module reset 👌')


class PiGPIO():
    def __init__(self):
        import gpiozero
        self.pwm_pins = [gpiozero.PWMOutputDevice(pin) for pin in [5, 6, 13, 19, 26]]  # GPIO numbering
        self.channels_number = len(self.pwm_pins)  # number of channels under control

    def setValue(self, channel, percentage):
        """set percentage of pwm duty cycle

        Parameters
        ----------
        channel : int
            channel 0-15
        percentage : float
            0-1
        """
        self.pwm_pins[channel].value = percentage

    def reset_all(self):
        for c in range(self.channels_number):
            self.setValue(c, 0)
        print('⚙️  PWM module reset 👌')


class PWM(PCA9685, PiGPIO):
    def __init__(self, address=0x40, bus=8, debug=False):
        if platform.machine() == 'aarch64':
            self.module = PCA9685
            PCA9685.__init__(self, address=address, bus=bus, debug=debug)
        elif platform.machine() == 'armv7l':
            self.module = PiGPIO
            PiGPIO.__init__(self)

    def setValue(self, channel, percentage):
        return self.module.setValue(self, channel, percentage)


if __name__ == "__main__" and sys.argv[1] == 'reset':
    """reset pwm duty cycle of all channels to zero. run with:
      >>> python pwm.py reset
    """
    pwm = PWM()
    pwm.reset_all()

if __name__ == '__main__' and sys.argv[1] == 'manual':
    """set duty cycle of specific pwm channel manually. run with:
      >>> python pwm.py manual
    """
    pwm = PWM()
    print('- input format: [channel],[duty]\n- input q to quit')
    while True:
        command = input('input: ')
        if command == 'q':
            break
        elif command == '':
            continue
        eval(f'pwm.setValue({command})')
