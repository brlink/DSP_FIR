import numpy as np
import matplotlib.pyplot as plt


"""
ring buffer
"""
class FIR_filter:
    def __init__(self, _coefficients):
        self._coefficients = _coefficients
        # The length of buffer is the total number of taps
        self.buffer = np.zeros(len(_coefficients))
        self.buffer_offset = 0

        if(int(len(self._coefficients)) != 0):
            print("The ntap is:", len(self._coefficients))
        else:
            print("The filter has no coefficients. nTaps = 0.")

    def dofilter(self, v):
        self.buffer[self.buffer_offset] = v
        # Initialize the output result to 0
        result = 0
        # Initialize the buff_val
        # buffer_val: store new data, start of the delay line
        buffer_val = self.buffer_offset
        loop = self.buffer_offset

        for i in range(loop + 1):
            # self._coefficients[i]: impulse response
            result += self.buffer[buffer_val] * self._coefficients[i]
            # Changing and decrease buffer_val
            buffer_val = buffer_val - 1

        buffer_val = len(self._coefficients) - 1

        while (self.buffer_offset < buffer_val):
            result += self.buffer[buffer_val] * self._coefficients[loop+1]

            # Changing and increase the offset_0
            loop = loop + 1
            buffer_val = buffer_val - 1

        # Judge whether has extra space or not
        if ((buffer_val + 1) < len(self._coefficients)):
            self.buffer_offset = self.buffer_offset + 1

        else:
            self.buffer_offset = 0

        return result


def unittest():
    print('Test 1:')
    a = [3, 4, 5]
    h = [2, 3]
    print("The signal is: ", a)
    print("The coefficient is: ", h)
    output = np.zeros(len(a))
    fir_filter = FIR_filter(h)
    print(fir_filter)
    for index in range(len(a)):
        output[index] = fir_filter.dofilter(a[index])
        # The expectation of output result is 6, 17, 22. The print result is same with my expectation
        print("The result: ", output[index])

    # Now, I use another coefficient to verify it.
    print("\nTest 2:")
    a = [3, 4, 5]
    h = [2, 3, 4]
    print("The signal is: ", a)
    print("The coefficient is: ", h)
    output = np.zeros(len(a))
    fir_filter = FIR_filter(h)

    for index in range(len(a)):
        output[index] = fir_filter.dofilter(a[index])
        # The expectation of output result is 6, 17, 34. The print result is same with my expectation
        print("The result: ", output[index])

    # Omega vector
    w = np.linspace(-np.pi, np.pi, 100)

    # Amplitude response
    y = abs(np.cos(0.5 * w))

    plt.figure()
    plt.plot(w, y)
    plt.ylabel('sample')
    plt.xlabel('Amplitude')
    plt.savefig('Cos sample.svg', dpi=900, format='svg')
    plt.show()

    print('Test 3:')
    # Frequency response by sending a test signal into filter
    # Create noise
    # # Test signal is noisy (contains all frequencies)
    x = np.random.normal(0, 1, 1000)

    # Create averaging filter
    # Coefficients: 1/2
    h = np.array([1/2, 1/2])

    output = np.zeros(len(x))

    fir_filter = FIR_filter(h)
    for index in range(len(x)):
        output[index] = fir_filter.dofilter(x[index])

    """
        For the frequency domain of the random noise, we could see the trend of this plot is similar 
        with the trend of y = cos0.5*w. So, that is our expectation
    """

    fft_output = np.fft.fft(output)
    plt.figure()
    plt.plot(abs(fft_output))
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.title('Frequency domain - averaging filter 1/2 (Noise)')
    plt.savefig('Random noise coe_1.svg', dpi=900, format='svg')
    plt.show()

    """
        For the frequency domain of the array (1 to 10), we could see the trend of this plot 
        is similar with the trend of y = cos0.5*w. So, that is our expectation
    """

    print('Test 4:')
    array = range(1, 10)
    fir_filter = FIR_filter(h)
    for index in range(len(array)):
        output[index] = fir_filter.dofilter(array[index])

    fft_output = np.fft.fft(output)
    plt.figure()
    plt.plot(abs(fft_output))
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.title('Frequency domain - averaging filter 1/2 (Array1)')
    plt.savefig('Array coe_1.svg', dpi=900, format='svg')
    plt.show()

    print('Test 5:')
    """
        For the frequency domain of the random noisy and different coefficients, we could see the trend 
        of this plot is similar with our expectation
    """
    #Create averaging filter
    # Coefficients: 1/4
    h = np.array([1/4, 1/4, 1/4, 1/4])

    output = np.zeros(len(x))

    fir_filter = FIR_filter(h)
    for index in range(len(x)):
        output[index] = fir_filter.dofilter(x[index])

    fft_output = np.fft.fft(output)
    plt.figure()
    plt.plot(abs(fft_output))
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.title('Frequency domain - averaging filter 1/4 (Noise)')
    plt.savefig('Random noise coe_2.svg', dpi=900, format='svg')
    plt.show()

    print('Test 6：')
    """
        For the frequency domain of the array and different coefficients, we could see the trend 
        of this plot is similar with our expectation
    """
    array = range(1, 10)
    fir_filter = FIR_filter(h)
    for index in range(len(array)):
        output[index] = fir_filter.dofilter(array[index])

    fft_output = np.fft.fft(output)
    plt.figure()
    plt.plot(abs(fft_output))
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.title('Frequency domain - averaging filter 1/4 (Array1)')
    plt.savefig('Array coe_2.svg', dpi=900, format='svg')
    plt.show()


if __name__ == "__main__":
    unittest()
