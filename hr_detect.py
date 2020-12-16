import matplotlib.pyplot as plt
import numpy as np
import fir_filter
from ecg_gudb_database import GUDb


def find_peak(_det, lower, upper, period):
    # find possible point which is greater than points between it and in a period
    possible_point = []
    for i in range(0, len(_det) - 1):
        if _det[i] > _det[i + 1] and _det[i] > _det[i - 1] and lower < _det[i] < upper:
            possible_point.append(_det[i])

    # turn the point into sample index
    axis = []
    for point in possible_point:
        location = np.where(_det == point)
        axis.append([location[0][0], point])

    # set a threshold to detect wrong peaks
    r_detections = [axis[0]]
    for i in range(1, len(axis)):
        if abs(axis[i][0] - r_detections[-1][0]) < period:
            if axis[i][1] > r_detections[-1][1]:
                r_detections.pop()
                r_detections.append(axis[i])
        else:
            r_detections.append(axis[i])

    # format into proper type
    results = []
    for detections in r_detections:
        results.append(detections[0])
    return results


def remove_50dc():
    # delta F = taps/ sampleRate = 500 / 250 = 2
    # Number of taps
    taps = 250
    _50_start = 45
    _50_end = 55
    # DC: Selecting less than 5 and more than 245
    _dc_start = 5
    _dc_end = 245

    # _50_start/sampleRate: normalized frequency
    # Then scaled to the length of taps
    mapping_50_start = int(_50_start / ecg_rate * taps)
    mapping_50_end = int(_50_end / ecg_rate * taps)
    mapping_dc_start = int(_dc_start / ecg_rate * taps)
    mapping_dc_end = int(_dc_end / ecg_rate * taps)

    # Initialize the array removed_data
    removed_data = np.ones(taps)
    # Make the 50 Hz([45: 55]) be zero
    removed_data[mapping_50_start: mapping_50_end + 1] = 0
    removed_data[taps - mapping_50_end: taps - mapping_50_start + 1] = 0
    # Make the DC be zero
    removed_data[0: mapping_dc_start] = 0
    removed_data[mapping_dc_end: taps] = 0

    # Convert the frequency to sample-amplitude
    removed_data = np.fft.ifft(removed_data)
    removed_data = np.real(removed_data)
    coefficient = np.zeros(taps)
    coefficient[0: int(taps / 2)] = removed_data[int(taps / 2): taps]
    coefficient[int(taps / 2): taps] = removed_data[0: int(taps / 2)]
    coefficient = coefficient * np.hamming(taps)

    return coefficient


def matched_filter(_dat):
    # 1.pre-filtering, remove 50Hz and DC
    _50dc = remove_50dc()
    _50dc_coeff = fir_filter.FIR_filter(_50dc)

    # Initialization
    filtered_dat = np.zeros(sample)

    for index in range(sample):
        # Remove 50Hz and DC
        filtered_dat[index] = _50dc_coeff.dofilter(_dat[index])

    # 2.get time-reversing template
    template = filtered_dat[1600: 1800]
    # time reverse
    reversed_temp = template[::-1]

    # 3.create FIR coefficient
    template_coeff = fir_filter.FIR_filter(reversed_temp)

    # Initialization
    det = np.zeros(sample)

    # 4.filtered ECG by with reversed template
    for index in range(sample):
        det[index] = template_coeff.dofilter(filtered_dat[index])

    # 5.square the result to improve the S/N
    det = det * det

    '''
    plt.figure()
    plt.plot(det)
    plt.title("Filtered heartbeat by matched_filter")
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.show()
    '''

    # 6.detect peaks and remove wrong peaks
    _peaks = find_peak(det, 0.2 * 1e-10, 1 * 1e-10, 100)
    print('R peaks of ECG_msc_matric_3.dat: ', _peaks)

    """
    fft_det = np.fft.fft(det)
    plt.figure()
    plt.plot(abs(fft_det))
    plt.title("Filtered heartbeat by matched_filter")
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.show()
    """


def filter_einthoven(_dat):
    # 1.pre-filtering, remove 50Hz and DC
    _50dc = remove_50dc()
    _50dc_coeff = fir_filter.FIR_filter(_50dc)

    # Initialization
    filtered_dat = np.zeros(sample)

    for index in range(sample):
        # Remove 50Hz and DC
        filtered_dat[index] = _50dc_coeff.dofilter(_dat[index])

    # 2.get time-reversing template
    template = filtered_dat[650: 800]
    # time reverse
    reversed_temp = template[::-1]

    # 3.create FIR coefficient
    template_coeff = fir_filter.FIR_filter(reversed_temp)

    # Initialization
    det = np.zeros(sample)

    # 4.filtered ECG by with reversed template
    for index in range(sample):
        det[index] = template_coeff.dofilter(filtered_dat[index])

    # 5.square the result to improve the S/N
    det = det * det

    # plt.figure()
    # plt.plot(det)
    # plt.title("Filtered heartbeat by matched_filter")
    # plt.xlabel('Time (s)')
    # plt.ylabel('Amplitude')
    # plt.show()

    # 6.detect peaks and remove wrong peaks
    _peaks = find_peak(det, 0.5 * 1e-11, 2 * 1e-11, 100)
    print('R peaks of Einthoven II walking record: ', str(_peaks))

    # 7.get momentary heart rate
    _heart_rate = []
    for i in range(len(_peaks) - 1):
        interval = _peaks[i + 1] - _peaks[i]
        beat = int(60 / (interval / einthoven_rate))
        _heart_rate.append(beat)

    print('Heart-rate: ', _heart_rate)

    time_axis = []
    for j in range(1, len(_peaks)):
        time_axis.append(round((_peaks[j] / einthoven_rate), 2))

    # 8.figure
    plt.figure()
    plt.ylim(0, 100)
    plt.plot(time_axis, _heart_rate, marker="o", mfc='r')
    plt.title('Momentary heart rate')
    plt.xlabel('Time(s)')
    plt.ylabel('Heart rate(times)')
    plt.savefig('momentary heart beat.svg', dpi=900, format='svg')
    plt.show()


if __name__ == "__main__":
    # load file
    dat_file = np.loadtxt("ECG_msc_matric_3.dat")
    # initialization
    sample = dat_file.shape[0]
    ecg_rate = 250

    matched_filter(dat_file)

    # get an Einthoven II walking recording
    _einthoven_dat = GUDb(19, 'walking').einthoven_II
    einthoven_rate = 250

    filter_einthoven(_einthoven_dat)
