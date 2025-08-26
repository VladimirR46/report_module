import numpy as np
import sys
import pyxdf
import logging
import json
import argparse
from pathlib import Path
from scipy.signal import firwin, butter, filtfilt, sosfiltfilt, welch

EXIT_SUCCESS = 0
EXIT_ERROR_NO_DATA = 1
EXIT_ERROR_PROCESSING = 2
EXIT_ERROR_JSON_WRITE = 3

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)04d %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

FRONTAL_CHANNELS      = np.array(['Fp1', 'Fp2', 'F7',  'Fz', 'F8'])
CENTRAL_L_CHANNELS    = np.array(['T3', 'C3', 'P3', 'F3'])
CENTRAL_R_CHANNELS    = np.array(['T4', 'C4', 'P4', 'F4'])
OCCIPITAL_CHANNELS    = np.array(['O1', 'O2','Pz','T6','T5'])

REGIONS = {
    'frontal': FRONTAL_CHANNELS,
    'central_l': CENTRAL_L_CHANNELS,
    'central_r': CENTRAL_R_CHANNELS,
    'occipital': OCCIPITAL_CHANNELS,
}

def bandpass_filter(data, sfreq, l_freq, h_freq, method='fir'):
    if method == 'fir':
        l_trans = min(max(0.25 * l_freq, 2.0), l_freq)
        h_trans = min(max(0.25 * h_freq, 2.0), sfreq/2 - h_freq)
        trans_bandwidth = min(l_trans, h_trans)
        n_taps = int(3.3 * sfreq / trans_bandwidth) | 1
        filt = firwin(n_taps, [l_freq, h_freq], fs=sfreq, pass_zero='bandpass')
        return filtfilt(filt, 1, data, axis=-1)
    else:
        nyq = sfreq / 2
        sos = butter(4, [l_freq/nyq, h_freq/nyq], 'band', output='sos')
        return sosfiltfilt(sos, data, axis=-1)

def psd_welch(data, sfreq, fmin=0.0, fmax=np.inf):
    nperseg = int(min(2 * sfreq, data.shape[-1]))
    nfft = max(nperseg, int(4 * sfreq))
    freqs, psds = welch(data, sfreq, nperseg=nperseg,nfft=nfft, noverlap=nperseg//2)
    freq_mask = (freqs >= fmin) & (freqs <= fmax)
    psds = psds[:, freq_mask] if psds.ndim > 1 else psds[freq_mask]
    return psds, freqs[freq_mask]

def power_band(psds, freqs, bands, power='abs'):
    psd = psds.copy()
    if power == 'rel':
        psd /= np.sum(psd, axis=-1, keepdims=True)
    band_powers = {}
    for band, (fmin, fmax) in bands.items():
        mask = (freqs >= fmin) & (freqs < fmax)
        band_powers[band] = np.mean(psd[:, mask], axis=1)
    return band_powers

def get_stream_id(streams, stream_name=None, stream_type=None):
    """Find stream ID by name and/or stream type."""
    for stream_id, stream in streams.items():
        info = stream['info']
        if stream_name is not None and info['name'][0] != stream_name:
            continue
        if stream_type is not None and info['type'][0] != stream_type:
            continue
        return stream_id
    criteria = []
    if stream_name: criteria.append(f"name='{stream_name}'")
    if stream_type: criteria.append(f"type='{stream_type}'")
    logger.error(f"Stream with {' and '.join(criteria)} not found")
    return None

class ProtocolData:
    def __init__(self, directory):
        self.directory = directory
        self.task, self.streams = self.load_data()
        self.eeg_stream = self.get_eeg_stream()
        self.eeg_data, self.ch_list = self.load_eeg_data()
        self.events = self.get_events()
        self.patient = self.task['patient']
        self.srate = int(float(self.eeg_stream["info"]["nominal_srate"][0]))

    def load_data(self):
        with open(self.directory / 'Task.json', encoding='utf-8') as file:
            task = json.load(file)
        streams, _ = pyxdf.load_xdf(self.directory / 'data.xdf', synchronize_clocks=True, dejitter_timestamps=True)
        streams = {stream["info"]["stream_id"]: stream for stream in streams}
        return task, streams

    def get_eeg_stream(self):
        eeg_id = get_stream_id(self.streams, stream_type='EEG')
        return self.streams[eeg_id]

    def load_eeg_data(self):
        ch_list, units = [], []
        for ch in self.eeg_stream["info"]["desc"][0]["channels"][0]["channel"]:
            ch_list.append(str(ch["label"][0]))
            units.append(ch["unit"][0] if ch["unit"] else "NA")
        time_series = self.eeg_stream["time_series"]
        #microvolts = ("microvolt", "microvolts", "µV", "μV", "uV")
        #scale = np.array([1e-6 if u in microvolts else 1 for u in units])
        #time_series_scaled = (time_series * scale).T
        time_series_scaled = time_series.T
        return time_series_scaled, ch_list

    def get_events(self):
        events = self.task['events']
        samples = self.task['samples']
        first_time = self.eeg_stream["time_stamps"][0]
        event_dict = {'source': [], 'event_name': [], 'sample_type': [], 'trial_type': [], 'block_type': [],
                      'block_id': [], 'trial_id': [], 'event_id': [], 'item_id': [], 'sample_id': [],
                      'trigger_code': [],
                      'time': [], 'index': [], 'duration': []}
        for event in events:
            sample = samples[event['sample_id']]
            event_dict['source'].append(event['source'])
            event_dict['event_name'].append(event['event_name'])
            event_dict['sample_type'].append(sample['sample_type'])
            event_dict['trial_type'].append(sample['trial_type'])
            event_dict['block_type'].append(sample['block_type'])
            event_dict['block_id'].append(sample['block_id'])
            event_dict['trial_id'].append(sample['trial_id'])
            event_dict['event_id'].append(event['event_id'])
            event_dict['item_id'].append(event['item_id'])
            event_dict['sample_id'].append(event['sample_id'])
            event_dict['trigger_code'].append(sample['trigger_code'])
            event_dict['time'].append(event['time'] - first_time)
            event_dict['duration'].append(float(sample['duration']) if event['event_name'] == 'show' else 0.0)
        for key in event_dict:
            event_dict[key] = np.array(event_dict[key])
        return event_dict

    def pick_channels(self, names):
        indices = []
        missing_channels = []
        for name in names:
            if name in self.ch_list:
                indices.append(self.ch_list.index(name))
            else:
                missing_channels.append(name)
        if missing_channels:
            logger.error(f"Channels not found: {missing_channels}")
            return []
        return indices

def truncate(val, decimals=3):
    sig = max(0, 4 - len(str(int(val))))
    return format(val , '.%df' % sig)

def background_process(protocol):
    b_idxs = np.where(protocol.events['sample_type'] == 'Background')[0]
    if len(b_idxs) != 2:
        logger.warning("number of backgrounds is not equal to 2")
        return None
    b_time = protocol.events['time'][b_idxs]
    b_dur = protocol.events['duration'][b_idxs]
    b_start = np.round(b_time * protocol.srate).astype(int)
    b_end = np.round((b_time + b_dur) * protocol.srate).astype(int)

    bands = {'delta': (1, 4), 'theta': (4, 8), 'alpha': (8, 13), 'beta': (13, 30)}
    stages = ['start', 'end']
    out = {k: {} for k in stages}

    for i, (start, end) in enumerate(zip(b_start, b_end)):
        stage = stages[i]
        data = protocol.eeg_data[:, start:end]

        psds, freqs = psd_welch(data, sfreq=protocol.srate, fmin=0.5, fmax=30)
        power_abs = power_band(psds, freqs, bands, power='abs')
        #power_rel = {
        #    band: power / sum(power_abs.values())
        #    for band, power in power_abs.items()
        #}
        power_rel = power_band(psds, freqs, bands, power='rel')

        for region_name, ch_list in REGIONS.items():
            ch_idx = protocol.pick_channels(ch_list)
            reg = out[stage].setdefault(region_name, {})

            for band_name in bands.keys():
                v_abs = truncate(np.mean(power_abs[band_name][ch_idx]), 3)
                v_rel = truncate(np.mean(power_rel[band_name][ch_idx]), 3)
                reg[band_name] = {"abs": v_abs, "rel": v_rel}
    return out

def rest_stim_bands(events, hand, srate):
    rest = np.where(
        (events['sample_type'] == 'Rest') &
        (events['trial_type'] == hand+'/hand') &
        (events['event_name'] == 'show')
    )[0]
    rest_time = events['time'][rest]
    stim = np.where(
        (np.isin(events['sample_type'], ['Point', 'Image'])) &
        (events['trial_type'] == hand+'/hand') &
        (events['event_name'] == 'show')
    )[0]
    stim_time = events['time'][stim]

    rest_start = np.round(rest_time * srate).astype(int)
    rest_end = np.round((rest_time+events['duration'][rest]) * srate).astype(int)
    rest = np.column_stack((rest_start, rest_end))

    stim_start = np.round(stim_time * srate).astype(int)
    stim_end = np.round((stim_time+events['duration'][stim]) * srate).astype(int)
    stim = np.column_stack((stim_start, stim_end))
    return rest, stim

def calc_trials(protocol, rest, stim):
    r"""
    Returns
    -------
    trials : ndarray
        (trials_n, f_bands, ch_n)
    """
    trials = []
    for (r_start, r_end), (s_start, s_end) in zip(rest, stim):
        rest_data = protocol.eeg_data[:, r_start:r_end]
        stim_data = protocol.eeg_data[:, s_start:s_end]

        rest_psds, rest_freqs = psd_welch(rest_data, sfreq=protocol.srate, fmin=1, fmax=30)
        stim_psds, stim_freqs = psd_welch(stim_data, sfreq=protocol.srate, fmin=1, fmax=30)

        def power_band(psds, freqs, bands):
            band_powers = {}
            for band, (fmin, fmax) in bands.items():
                mask = (freqs >= fmin) & (freqs < fmax)
                band_powers[band] = np.mean(psds[:, mask], axis=1)
            return band_powers

        bands = {'mu': (8, 13),'beta': (13, 30)}
        rest_bands = power_band(rest_psds, rest_freqs, bands)
        stim_bands = power_band(stim_psds, stim_freqs, bands)

        ERD_mu = (rest_bands['mu'] - stim_bands['mu']) / rest_bands['mu'] * 100
        ERD_beta = (rest_bands['beta'] - stim_bands['beta']) / rest_bands['beta'] * 100
        ERD_mean = (ERD_mu + ERD_beta) / 2
        trials.append([ERD_mu, ERD_beta, ERD_mean])
    return np.array(trials)

def erd_process(protocol):
    rest_l, stim_l = rest_stim_bands(protocol.events, 'left', protocol.srate)
    rest_r, stim_r = rest_stim_bands(protocol.events, 'right', protocol.srate)

    l_trials = calc_trials(protocol, rest_l, stim_l)
    r_trials = calc_trials(protocol, rest_r, stim_r)

    # Make Json
    out = {}
    for hand, data in [('left', l_trials), ('right', r_trials)]:
        ch_dict = {}

        ch_names = ['c3', 'c4', 'cz']
        picked = protocol.pick_channels(['C3', 'C4', 'Cz'])
        for ch_name, ch_idx in zip(ch_names, picked):
            ch_data = data[:, :, ch_idx]
            band_dict = {}
            for band, band_idx in (('mu', 0), ('beta', 1), ('erd', 2)):
                band_data = ch_data[:, band_idx]
                band_dict[band] = {'mean': truncate(np.mean(band_data)), 'max': truncate(np.max(band_data))}
            ch_dict[ch_name] = band_dict
        out[hand] = ch_dict
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path', help='Path to patient folder')
    args = parser.parse_args()

    directory = Path(args.data_path)
    if not directory.exists():
        logger.error(f"Error: path {args.data_path} does not exist")
        return EXIT_ERROR_NO_DATA

    protocol = ProtocolData(directory)
    protocol.eeg_data = bandpass_filter(protocol.eeg_data, protocol.srate, l_freq=1, h_freq=40, method='iir')

    # Process Background
    background_result = background_process(protocol)
    # Process ERD
    erd_result = erd_process(protocol)

    result = background_result
    result['imagin'] = erd_result

    if result is None:
        logger.error("No data to process")
        return EXIT_ERROR_PROCESSING

    with open(protocol.directory / 'report_data.json', "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    logger.info("Report successfully created")

    return EXIT_SUCCESS

if __name__ == "__main__":
    sys.exit(main())