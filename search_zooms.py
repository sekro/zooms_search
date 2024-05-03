"""
Script to batch search Bruker peaklist.xml files agains ZooMS marker data
@author Sebastian Krossa, MR Core Facility, ISB, MH, NTNU, Norway, 2024
sebastian.krossa@ntnu.no
"""

import pandas as pd
import numbers
import operator
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
import xml.etree.ElementTree as ET
import argparse
import sys
import datetime

# globa config & def section

matplotlib.use("pdf")

g_DEBUG = False

defaults = {
    'min_hits': 3,
    'max_match_error': 1.2,
    'bruker_peaklist_filename': 'peaklist.xml'
}

g_pf_ops = {
    '==': operator.eq,
    '<': operator.lt,
    '<=': operator.le,
    '>': operator.gt,
    '>=': operator.ge}


def read_fa_peaklist(pl_path, sample_name=None):
    _data = []
    tree = ET.parse(pl_path)
    root = tree.getroot()
    for idx, pk in enumerate(root):
        _row = {}
        for _d in pk:
            _row[_d.tag] = float(_d.text)
        if sample_name:
            pk_name = 'sample:{}_peak:{}'.format(sample_name, idx + 1)
        else:
            pk_name = 'sample:UKNOWN_peak:{}'.format(idx + 1)
        _data.append(pd.Series(data=_row, name=pk_name))
    if len(_data) > 0:
        return pd.concat(_data, axis=1, verify_integrity=True).T
    else:
        return None


def get_sample_name(peaklist_path, max_path_up=3):
    _sn = None
    if os.path.exists(os.path.join(peaklist_path, 'proc')):
        with open(os.path.join(peaklist_path, 'proc'), 'r') as _f:
            for _line in _f:
                if '##$SMPNAM=' in _line:
                    _sn = str(_line).split('##$SMPNAM=')[1].strip().replace('\\', '-').replace('/', '_')[1:-1]
                    break
    elif os.path.exists(os.path.join(peaklist_path, 'procs')):
        with open(os.path.join(peaklist_path, 'procs'), 'r') as _f:
            for _line in _f:
                if '##$SMPNAM=' in _line:
                    _sn = str(_line).split('##$SMPNAM=')[1].strip().replace('\\', '-').replace('/', '_')[1:-1]
                    break
    else:
        _tail = os.path.splitdrive(peaklist_path)[1]
        _join_these = []
        _collect = False
        _loops = 0
        while len(_tail) > 0:
            if _tail == '\\':
                break
            _tail, _head = os.path.split(_tail)
            if _collect:
                _join_these.append(_head)
            if _head == '1SRef':
                _collect = True
            if len(_join_these) >= max_path_up:
                break
            if _loops >= 10:
                break
            _loops += 1
        _sn = '-'.join(reversed(_join_these))
    if len(_sn) == 0:
        _sn = 'Unknown-{}'.format(
            os.path.splitdrive(peaklist_path)[1].replace('.', '').replace('\\', '-').replace('/', '_'))
    return _sn


def read_zooms_db(zooms_file):
    _raw_zooms_db = pd.read_excel(zooms_file, sheet_name=None)
    _filtered_zooms_db = {}
    for _n, _df in _raw_zooms_db.items():
        if not 'Reference' in _n:
            _label_row = _df[_df.iloc[:, 0] == 'Order'].index[0]
            _rename_cols = _df.iloc[_label_row, :].to_dict()
            _mult_idx_cols = _df.iloc[_label_row, 0:4].to_list()
            _filtered_zooms_db[_n] = _df.iloc[_label_row + 1:, :].rename(columns=_rename_cols).reset_index(
                drop=True).set_index(_mult_idx_cols)
    _filtered_zooms_db['info_cols'] = ['Reference', 'LC-MS/MS verified?', 'Notes']
    return _filtered_zooms_db


def clean_zooms_entry(raw_zooms_entry):
    """
    Some entries in the ZooMS excel file have multiple values with / or , per column
    also filter out no entries
    :param raw_zooms_entry:
    :return: filtered zoom entries as float np array
    """
    _filtered_entry = []
    for _e in raw_zooms_entry:
        if isinstance(_e, numbers.Number):
            if not np.isnan(_e):
                _filtered_entry.append(_e)
        elif isinstance(_e, str):
            _se = None
            if '/' in _e:
                _se = _e.split('/')
            elif ',' in _e:
                _se = _e.split(',')
            if _se is not None:
                for _ese in _se:
                    _ese = _ese.strip()
                    if _ese.replace('.', '', 1).isdigit():
                        _filtered_entry.append(float(_ese))
    return np.round(_filtered_entry, 1)


def plot_hits_centroid_spec(hit, peak_table, figsize=(10, 4), sample_name=None):
    fig, ax = plt.subplots(figsize=figsize)
    _mzs = []
    _ints = []
    _mzs_h = []
    _ints_h = []
    if hit is not None:
        for _mz, _int in zip(peak_table['mass'], peak_table['absi']):
            if np.round(_mz, 1) in hit['matched_peaks']:
                _mzs_h.append(_mz)
                _ints_h.append(_int)
            else:
                _mzs.append(_mz)
                _ints.append(_int)
    else:
        _mzs = peak_table['mass'].to_list()
        _ints = peak_table['absi'].to_list()
    ax.stem(_mzs, _ints, markerfmt='', linefmt='k-', basefmt='k-')
    if hit is not None:
        ax.stem(_mzs_h, _ints_h, markerfmt='', linefmt='r-', label='matched peaks', basefmt='k-')
        ax.legend()
    ax.set_ylabel('Intensity')
    ax.set_xlabel('m/z')
    _hn = ''
    if hit is not None:
        for _e in hit.name:
            _hn += '{}, '.format(_e)
        ax.set_title('Matched peaks, sample: {}, ID: {}'.format(hit['sample_id'], _hn.strip()[0:-1]))
    else:
        ax.set_title('No matched peaks, sample: {}'.format(sample_name))
    return fig, ax


def gen_rename_results_idx_cols_dict(zooms_db):
    _retd = {}
    for _i, _idx in enumerate(zooms_db[list(zooms_db.keys())[0]].index.names):
        _retd['level_{}'.format(_i)] = _idx
    return _retd


def match_peaks2entry(pks, entry, error_da=1.2):
    _matches_entry = []
    _matched_pks = []
    for _pk in pks:
        for _epk in entry:
            if (_epk - error_da) <= _pk <= (_epk + error_da):
                _matches_entry.append(_epk)
                _matched_pks.append(_pk)
                break
    return _matches_entry, _matched_pks



def search_zooms(peaks_table, zooms_db, max_error_da, peak_filters=None, sample_name=None, min_hits=None):
    """
    Search ZooMS db with peaks table
    :param peaks_table: peak table provided by function read_fa_peaklist
    :param zooms_db: the zooMS db provided by function read_zooms_db
    :param peak_filters: multiple values to filter the peaktable as
                         Dict -> {'peak_table_col': {'op': '>=', 'value': 0.6}}
    :param sample_name: name of sample, str
    :param min_hits: integer, minimum number of matched peaks to count as hit
    :return: Dict with hits
    """
    _hits = {}
    _pf_ops = g_pf_ops
    if sample_name is None:
        _current_sample = 'A1'
    else:
        _current_sample = sample_name
    if min_hits is None:
        min_hits = defaults['min_hits']
    _filtered_table = peaks_table
    if peak_filters is not None:
        _info = 'using only peaks with '
        for _fn, _fv in peak_filters.items():
            if _fn in _filtered_table.columns and _fv['op'] in _pf_ops:
                _filtered_table = _filtered_table[_pf_ops[_fv['op']](_filtered_table[_fn], _fv['value'])]
                _info += '{} {} {}, '.format(_fn, _fv['op'], _fv['value'])
    else:
        _info = 'using all peaks'
    _info = '{}, allowing a max error of +/- {} Da during peak matching'.format(_info, max_error_da)
    _pl = np.round(_filtered_table['mass'].to_list(), 1)
    for _s, _df in zooms_db.items():
        if not _s == 'info_cols':
            _c = _df.columns.to_list()
            _c = [_e for _e in _c if _e not in zooms_db['info_cols']]
            if g_DEBUG:
                print('DEBUG: Searching in {} using columns: {}'.format(_s, _c))
            for _ri, _rd in _df[_c].iterrows():
                _matched_entries, _matched_peaks = match_peaks2entry(pks=_pl, entry=clean_zooms_entry(_rd.to_list()), error_da=max_error_da)
                if len(_matched_peaks) >= min_hits:
                    _hits[_ri] = {
                        'n_matched_peaks': len(_matched_peaks),
                        'matched_peaks': _matched_peaks,
                        'matched_zooms_peaks': _matched_entries,
                        'sample_id': _current_sample,
                        'info': _info}
    return _hits


def df2excel_wrapper(df, outpath, force_save=True):
    try:
        df.to_excel(outpath)
    except PermissionError:
        if force_save:
            _b, _ext = os.path.splitext(outpath)
            _nout = '{}_{date:%Y-%m-%d_%H-%M-%S}{ext}'.format(_b, date=datetime.datetime.now(), ext=_ext)
            df.to_excel(_nout)


def gen_peakfilter_dict(filter_args):
    _rd = {}
    for _fa in filter_args:
        for _op in g_pf_ops.keys():
            if _op in _fa:
                _sr = _fa.split(_op)
                if len(_sr) == 2 and not '=' in _sr[1]:
                    _rd[_sr[0]] = {'op': _op, 'value': float(_sr[1])}
    return _rd


def main(peak_file, out_base_path, sample_name, zooms_db, max_error_da,
         peak_filters=None, min_hits=None):
    _out_path = os.path.join(out_base_path, sample_name)
    if not os.path.exists(_out_path):
        os.mkdir(_out_path)
    _pk_table = read_fa_peaklist(pl_path=peak_file, sample_name=sample_name)
    if _pk_table is not None:
        df2excel_wrapper(_pk_table, os.path.join(_out_path, 'peak_table_{}.xlsx'.format(sample_name)), force_save=True)
        _hits = search_zooms(peaks_table=_pk_table, zooms_db=zooms_db, max_error_da=max_error_da,
                             peak_filters=peak_filters, sample_name=sample_name, min_hits=min_hits)
    else:
        print('No peaks found in current sample {}'.format(sample_name))
        return None, None
    if len(_hits) > 0:
        _hits_df = pd.DataFrame(data=_hits).T.sort_values('n_matched_peaks', ascending=False)
        df2excel_wrapper(_hits_df, os.path.join(_out_path, 'hits_table_{}.xlsx'.format(sample_name)), force_save=True)
        fig, ax = plot_hits_centroid_spec(hit=_hits_df.iloc[0, :], peak_table=_pk_table)
        fig.savefig(os.path.join(_out_path, 'top_hit_centroid_plot.pdf'), dpi=600, format='pdf', bbox_inches='tight')
        plt.close(fig=fig)
        if len(_hits_df[_hits_df['n_matched_peaks'] == _hits_df['n_matched_peaks'].max()]) > 1:
            return _hits_df.iloc[0, :], _hits_df.iloc[1, :]
        else:
            return _hits_df.iloc[0, :], None
    else:
        print('No hits found for sample {}'.format(sample_name))
        fig, ax = plot_hits_centroid_spec(hit=None, peak_table=_pk_table, sample_name=sample_name)
        fig.savefig(os.path.join(_out_path, 'No_hit_centroid_plot.pdf'), dpi=600, format='pdf', bbox_inches='tight')
        plt.close(fig=fig)
        return None, None


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='search_zooms - Script to batch search Bruker peaklist.xml files '
                                                 'agains ZooMS marker data')
    parser.add_argument("input", help="folder with input files, expecting Bruker flexControl format, peaklist.xml")
    parser.add_argument("output", help="folder for output files")
    parser.add_argument("zooms_db", help="path to the ZooMS Marker excel file")
    parser.add_argument("--min_matched_peaks",
                        help="Minimum number of matched peaks to accept as potential hit - default {}".format(
                            defaults['min_hits']),
                        default=defaults['min_hits'])
    parser.add_argument("--max_match_error",
                        help="Max error in Da to accept a peak match - default {}".format(
                            defaults['max_match_error']),
                        default=defaults['max_match_error'])
    parser.add_argument("--peak_filter", nargs='+',
                        help="Add one or multiple filters (separated by whitespace) to filter peaklist.xml before "
                             "searching for hits. Example: --filter goodn2>=0.7 s2n>=5")
    parser.add_argument("--bruker_peaklist_filename",
                        help="Name of the Bruker peak list files - default {}".format(
                            defaults['bruker_peaklist_filename']),
                        default=defaults['bruker_peaklist_filename'])
    args = parser.parse_args()
    # check and make out folder
    if not os.path.exists(args.output):
        os.mkdir(args.output)
    if os.path.exists(args.input) and os.path.exists(args.zooms_db):
        if args.peak_filter is not None:
            _peak_filters = gen_peakfilter_dict(args.peak_filter)
            print(_peak_filters)
        else:
            _peak_filters = None
        _zdb = read_zooms_db(zooms_file=args.zooms_db)
        _top_hits_list = []
        input_files = {}
        for current_dir, dirs, files in os.walk(args.input):
            for file in files:
                if file == args.bruker_peaklist_filename:
                    _current_sample_pos = get_sample_name(current_dir)
                    if not 'LIFT' in _current_sample_pos:
                        input_files[_current_sample_pos] = os.path.join(current_dir, file)
                        print('Found peak list file {} for sample {}'.format(os.path.join(current_dir, file),
                                                                             _current_sample_pos))
        for _sn, _fp in input_files.items():
            print('Working on sample: {}'.format(_sn))
            _current_top_hit, _current_next_hit = main(peak_file=_fp, out_base_path=args.output, sample_name=_sn,
                                                       zooms_db=_zdb, peak_filters=_peak_filters, max_error_da=float(args.max_match_error),
                                                       min_hits=int(args.min_matched_peaks))
            if _current_top_hit is not None:
                _top_hits_list.append(_current_top_hit)
            if _current_next_hit is not None:
                _current_next_hit['sample_id'] = '{} second hit'.format(_current_next_hit['sample_id'])
                _top_hits_list.append(_current_next_hit)
                print('Found at least 2 different hits with same number of matched peaks in sample {}'.format(_sn))

        if len(_top_hits_list) > 0:
            _df_top_hits = pd.concat(_top_hits_list, axis=1, verify_integrity=True).T.reset_index().set_index(
                'sample_id').rename(columns=gen_rename_results_idx_cols_dict(_zdb))
            df2excel_wrapper(_df_top_hits, os.path.join(args.output, 'all_top_hits.xlsx'), force_save=True)
        else:
            print('No hits found in any of the provided samples')
    else:
        print('Input folder does not exist - aborting')
        sys.exit(1)
    print('Finished search')
