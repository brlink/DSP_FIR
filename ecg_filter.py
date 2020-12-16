import matplotlib.pyplot as plt
import numpy as np
import fir_filter


def ECG_plot(dat_file0):
    # The total time of all heartbeats
    t = np.linspace(0, sample / sampleRate, sample)
    print("number of samples:")
    print(sample)
    print("sampling rate:")
    print(sampleRate)

    # time domain of initial dat file
    plt.figure()
    plt.plot(t, dat_file0)
    plt.title("Initial heartbeat")
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.savefig('Initial heartbeat.svg', dpi=900, format='svg')
    plt.show()

    # fft - frequency domain
    x_fft = np.fft.fft(dat_file0)

    # fft computing and normalization
    x_fft = x_fft/len(dat_file0)

    # Cut off half of frequency
    x_fft = x_fft[range(int(len(dat_file0)/2))]

    faxis = np.arange(len(dat_file0))/(len(dat_file0)/sampleRate)

    # Cut off half of faxis
    faxis = faxis[range(int(len(dat_file0)/2))]

    abs_x_fft = abs(x_fft)

    plt.figure()
    plt.plot(faxis, abs_x_fft)
    plt.title('Initial heartbeat frequency domain')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.ylim([0, 0.00004])
    plt.savefig('Initial heartbeat - frequency domain.svg', dpi=900, format='svg')
    plt.show()


def ECG_remove(dat_file0):
    # Get the coefficient
    h = Remove_50DC()
    # Using the fir filter to execute the coefficient
    _fir_filter = fir_filter.FIR_filter(h)

    # Initialize the array of filtered data
    filtered_dat = np.zeros(sample)

    # Divided the signal into length of samples
    # dat_file0 is used as the input signal
    for index in range(sample):
        # Call the function - dofilter to get the real time result
        filtered_dat[index] = _fir_filter.dofilter(dat_file0[index])

    # Plot the time domain of remove 50Hz data
    plt.figure()
    plt.plot(filtered_dat)
    plt.title("Remove 50Hz and DC")
    plt.savefig('Remove 50Hz and DC - time domain.svg', dpi=900, format='svg')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.show()

    # plot fft
    fft_filtered_dat = np.fft.fft(filtered_dat)
    # fft computing and normalization
    fft_filtered_dat = fft_filtered_dat/len(filtered_dat)
    # Cut off half of frequency
    fft_filtered_dat = fft_filtered_dat[range(int(len(filtered_dat)/2))]

    faxis = np.linspace(0, sampleRate, sample)
    # Cut off half of faxis
    faxis = faxis[range(int(len(filtered_dat)/2))]

    plt.figure()
    plt.plot(faxis, abs(fft_filtered_dat))
    plt.title('Remove 50Hz and DC')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.savefig('Remove 50Hz and DC - frequency domain.svg', dpi=900, format='svg')
    plt.show()

    # Create fir filtered dat file (remove 50Hz and DC)
    np.savetxt("shortecg.dat", filtered_dat, fmt='%f',delimiter=',')


def Remove_50DC():
    # delta F = M/ sampleRate = 500 / 250 = 2
    # Number of taps
    M = 150
    fre1_50 = 45
    fre2_50 = 55
    # DC: Selecting less than 5 and more than 245
    fre1_DC = 5
    fre2_DC = 245

    # fre1_50/sampleRate: normalized frequency
    # Then scaled to the length of M
    k1 = int(fre1_50 / sampleRate * M)
    k2 = int(fre2_50 / sampleRate * M)
    k3 = int(fre1_DC / sampleRate * M)
    k4 = int(fre2_DC / sampleRate * M)

    # Initialize the array x
    x = np.ones(M)
    # Make the 50 Hz([45: 55]) be zero
    x[k1: k2 + 1] = 0
    x[M - k2: M - k1 + 1] = 0
    # Make the DC be zero
    x[0: k3] = 0
    x[k4: M] = 0

    # Convert the frequency to sample-amplitude
    x = np.fft.ifft(x)
    x = np.real(x)
    h = np.zeros(M)
    h[0: int(M / 2)] = x[int(M / 2): M]
    h[int(M / 2): M] = x[0: int(M / 2)]
    h = h * np.hamming(M)

    return h


if __name__ == "__main__":
    dat_file0 = np.loadtxt("ECG_msc_matric_3.dat", dtype=np.float64)
    sample = dat_file0.shape[0]
    sampleRate = 250

    ECG_plot(dat_file0)

    ECG_remove(dat_file0)



