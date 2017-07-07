from scipy.signal import filtfilt, butter


def butterworth(x, order, freq=0.05):
    b, a = butter(order, freq)
    return filtfilt(b, a, x)
