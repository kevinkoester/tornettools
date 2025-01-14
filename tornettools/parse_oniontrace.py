import os
import logging
import datetime
import subprocess

from tornettools.util import which, cmdsplit, open_writeable_file, load_json_data, dump_json_data

from collections import defaultdict

def parse_oniontrace_logs(args):
    otracetools_exe = which('oniontracetools')

    if otracetools_exe == None:
        logging.warning("Cannot find oniontracetools in your PATH. Is your python venv active? Do you have oniontracetools installed?")
        logging.warning("Unable to parse oniontrace simulation data.")
        return

    cmd_str = f"{otracetools_exe} parse -m {args.nprocesses} -e 'oniontrace.*\.log' shadow.data/hosts"
    cmd = cmdsplit(cmd_str)

    datestr = datetime.datetime.now().strftime("%Y-%m-%d.%H:%M:%S")

    with open_writeable_file(f"{args.prefix}/oniontracetools.parse.{datestr}.log") as outf:
        logging.info("Parsing oniontrace log data with oniontracetools now...")
        comproc = subprocess.run(cmd, cwd=args.prefix, stdout=outf, stderr=subprocess.STDOUT)
        logging.info(f"oniontracetools returned code {comproc.returncode}")

    return comproc.returncode == 0

def extract_oniontrace_plot_data(args):
    json_path = f"{args.prefix}/oniontrace.analysis.json"

    if not os.path.exists(json_path):
        json_path += ".xz"

    if not os.path.exists(json_path):
        logging.warning(f"Unable to find oniontrace analysis data at {json_path}.")
        return

    data = load_json_data(json_path)

    # skip the first 20 minutes to allow the network to reach steady state
    startts, stopts = 1200, -1

    __extract_circuit_build_times(args, data, startts, stopts)
    __extract_relay_tput(args, data, startts, stopts)
    __extract_client_cicuit_list(args.prefix, data)
    __extract_client_stream_list(args.prefix, data)
    __extract_client_stream_dict(args.prefix, data)
    __extract_circuit_dict(args.prefix, data)
    __extract_circuit_bandwidth(args.prefix, data)

def __extract_circuit_build_times(args, data, startts, stopts):
    cbt = __get_perfclient_cbt(data, startts, stopts)
    outpath = f"{args.prefix}/tornet.plot.data/perfclient_circuit_build_time.json"
    dump_json_data(cbt, outpath, compress=False)

def __extract_relay_tput(args, data, startts, stopts):
    tput = __get_relay_tput(data, startts, stopts)
    outpath = f"{args.prefix}/tornet.plot.data/relay_goodput.json"
    dump_json_data(tput, outpath, compress=False)

def __extract_client_cicuit_list(prefix, data):
    circ_num = __get_client_circuit_list(data)
    outpath = f"{prefix}/tornet.plot.data/circuit_list.json"
    dump_json_data(circ_num, outpath, compress=False)
    
def __extract_client_stream_dict(prefix, data):
    stream_stats = __get_client_stream_dict(data)
    outpath = f"{prefix}/tornet.plot.data/stream_dict.json"
    dump_json_data(stream_stats, outpath, compress=False)

def __extract_client_stream_list(prefix, data):
    circ_num = __get_client_stream_list(data)
    outpath = f"{prefix}/tornet.plot.data/stream_list.json"
    dump_json_data(circ_num, outpath, compress=False)



def __extract_circuit_dict(prefix, data):
    circ_dict = __get_circuit_relay_dict(data)
    outpath = f"{prefix}/tornet.plot.data/circuit_dict.json"
    dump_json_data(circ_dict, outpath, compress=False)

def __extract_circuit_bandwidth(prefix, data):
    circuit_bandwidth = __get_circuit_bandwidth(data)
    outpath = f"{prefix}/tornet.plot.data/circuit_bandwidth.json"
    dump_json_data(circuit_bandwidth, outpath, compress=False)

def __get_circuit_bandwidth(data):
    node_types = ["markovclient", "perfclient"]
    circuit_dict = {x : defaultdict(lambda: defaultdict(lambda: defaultdict(int))) for x in node_types}
    for node_name, node_data in data["data"].items():
        for t in node_types:
            if t in node_name:
                for cid,circuit_data in node_data["oniontrace"]["circuit_bandwidth"].items():
                    for time, timed_data in circuit_data.items():
                        # TODO: group similar times
                        for val_name, val in timed_data.items():
                            circuit_dict[t]["{}_{}".format(node_name.replace(t, ""), cid)][int(time)-946684800][val_name] += int(val)
    return circuit_dict


def __get_circuit_relay_dict(data):
    node_types = ["markovclient", "perfclient"]
    circuit_dict = {x : defaultdict(list) for x in node_types}
    for node_name, node_data in data["data"].items():
        circuit_stats = {x : defaultdict(list) for x in node_types}
        for t in node_types:
            if t in node_name:
                circuit_relay_dict = node_data["oniontrace"]["circuit_relay_dict"]
                for cid, circuit_nodes in circuit_relay_dict.items():
                    circuit_dict[t]["{}_{}".format(node_name.replace(t, ""), cid)] = circuit_nodes
    return circuit_dict

def __get_client_circuit_list(data):
    node_types = ["markovclient", "perfclient"]
    circuit_stats = {x : defaultdict(list) for x in node_types}
    for node_name, node_data in data["data"].items():
        circuit_events = node_data["oniontrace"]["circuit_events"]
        for t in node_types:
            if t in node_name:
                for time in sorted(list(circuit_events["built"].keys()) + list(circuit_events["closed"].keys())):
                    if time in circuit_events["built"]:
                        for circ_id in circuit_events["built"][time]:
                            circuit_stats[t][int(time)-946684800].append("{}_{}".format(node_name.replace(t, ""), circ_id))
                    if time in circuit_events["closed"]:
                        for circ_id in circuit_events["closed"][time]:
                            circuit_stats[t][int(time)-946684800].append("-{}_{}".format(node_name.replace(t, ""), circ_id))
    return circuit_stats

def __get_client_stream_dict(data):
    node_types = ["markovclient", "perfclient"]
    stream_dict = {x : defaultdict(list) for x in node_types}
    for node_name, node_data in data["data"].items():
        circuit_stats = {x : defaultdict(list) for x in node_types}
        for t in node_types:
            if t in node_name:
                #self.stream_circ_dict[cid][state][second].append(stream_id)
                circuit_relay_dict = node_data["oniontrace"]["stream_circ_dict"]
                for cid, state_list in circuit_relay_dict.items():
                    stream_dict[t]["{}_{}".format(node_name.replace(t, ""), cid)] = state_list
    return stream_dict


def __get_client_stream_list(data):
    node_types = ["markovclient", "perfclient"]
    stream_stats = {x : defaultdict(list) for x in node_types}
    for node_name, node_data in data["data"].items():
        stream_events = node_data["oniontrace"]["stream_events"]
        for t in node_types:
            if t in node_name:
                node_num = node_name.replace(t, "")
                for time in sorted(list(stream_events["new"].keys()) + list(stream_events["closed"].keys())):
                    if time in stream_events["new"]:
                        for stream_id in stream_events["new"][time]:
                            stream_stats[t][int(time)-946684800].append("{}_{}".format(node_num, stream_id))
                    if time in stream_events["closed"]:
                        for stream_id in stream_events["closed"][time]:
                            stream_stats[t][int(time)-946684800].append("-{}_{}".format(node_num, stream_id))
    return stream_stats


def __get_perfclient_cbt(data, startts, stopts):
    perf_cbt = []

    # cbts can differ by microseconds
    resolution = 1.0/1000000.0

    if 'data' in data:
        for name in data['data']:
            if 'perfclient' not in name: continue

            circ = data['data'][name]['oniontrace']['circuit']
            key = 'build_time'
            if circ is None or key not in circ: continue

            cbt = circ[key]

            for secstr in cbt:
                sec = int(secstr)-946684800
                if sec >= startts and (stopts < 0 or sec < stopts):
                    for val in cbt[secstr]:
                        #item = [val, resolution]
                        item = val
                        perf_cbt.append(item)

    return perf_cbt

def __get_relay_tput(data, startts, stopts):
    net_tput_sec = {}

    # resolution in 1 byte
    resolution = 1

    if 'data' in data:
        for name in data['data']:
            if 'relay' not in name and '4uthority' not in name: continue

            bw = data['data'][name]['oniontrace']['bandwidth']
            key = 'bytes_written'
            if bw is None or key not in bw: continue

            tput = bw[key]

            for secstr in tput:
                sec = int(secstr)-946684800
                if sec >= startts and (stopts < 0 or sec < stopts):
                    bytes = int(tput[secstr])
                    net_tput_sec.setdefault(sec, 0)
                    net_tput_sec[sec] += bytes

    return net_tput_sec
