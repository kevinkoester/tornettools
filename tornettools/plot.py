import sys
import os
import logging

from itertools import cycle
import matplotlib.pyplot as pyplot
from matplotlib.ticker import FuncFormatter, MultipleLocator

from tornettools.util import load_json_data, find_matching_files_in_dir

from tornettools.plot_common import *
from tornettools.plot_tgen import plot_tgen
from tornettools.plot_oniontrace import plot_oniontrace

from collections import defaultdict
import operator
import copy
import pathlib
import random

from scipy.stats import entropy
import numpy as np
import math

#for debug only
from tornettools.debug import *


SIMTIME_ONE_SECOND = 1000000000.0

def run(args):
    logging.info("Plotting simulation results now")
    set_plot_options()

    logging.info("Plotting tornet comparisons")
    __plot_tornet(args)

    if args.plot_all:
        logging.info("Plotting all individual simulation results is requested")

        logging.info("Attempting to plot individual tgen results")
        plot_tgen(args)

        logging.info("Attempting to plot individual oniontrace results")
        plot_oniontrace(args)

    logging.info(f"Done plotting! PDF files are saved to {args.prefix}")

def __plot_tornet(args):
    args.pdfpages = PdfPages(f"{args.prefix}/tornet.plot.pages.pdf")

    logging.info("Loading tornet resource usage data")
    tornet_dbs = __load_tornet_datasets(args, "resource_usage.json")
    __plot_memory_usage(args, tornet_dbs)
    __plot_run_time(args, tornet_dbs)

    logging.info("Loading tornet resource usage data per node")
    tornet_dbs = __load_tornet_datasets(args, "resource_usage_node.json")
    __plot_node_memory_usage(args, tornet_dbs)
    logging.info("Loading tornet cpu usage data per node")
    tornet_dbs = __load_tornet_datasets(args, "cpu_node.json")
    __plot_node_cpu_usage(args, tornet_dbs)

    logging.info("Loading Tor metrics data")
    torperf_dbs = __load_torperf_datasets(args.tor_metrics_path)


    logging.info("Loading tornet relay goodput data")
    tornet_dbs = __load_tornet_datasets(args, "relay_goodput.json")
    net_scale = __get_simulated_network_scale(args)
    logging.info("Plotting relay goodput")
    __plot_relay_goodput(args, torperf_dbs, tornet_dbs, net_scale)

    logging.info("Loading tornet circuit build time data")
    tornet_dbs = __load_tornet_datasets(args, "perfclient_circuit_build_time.json")
    logging.info("Plotting circuit build times")
    __plot_circuit_build_time(args, torperf_dbs, tornet_dbs)

    logging.info("Loading tornet round trip time data")
    tornet_dbs = __load_tornet_datasets(args, "round_trip_time.json")
    logging.info("Plotting round trip times")
    __plot_round_trip_time(args, torperf_dbs, tornet_dbs)

    logging.info("Loading tornet transfer time data")
    tornet_dbs = __load_tornet_datasets(args, "time_to_last_byte_recv.json")
    logging.info("Plotting transfer times")
    __plot_transfer_time(args, torperf_dbs, tornet_dbs, "51200")
    __plot_transfer_time(args, torperf_dbs, tornet_dbs, "1048576")
    __plot_transfer_time(args, torperf_dbs, tornet_dbs, "5242880")

    logging.info("Loading tornet goodput data")
    tornet_dbs = __load_tornet_datasets(args, "perfclient_goodput.json")
    logging.info("Plotting client goodput")
    __plot_client_goodput(args, torperf_dbs, tornet_dbs)

    logging.info("Loading tornet transfer error rate data")
    tornet_dbs = __load_tornet_datasets(args, "error_rate.json")
    logging.info("Plotting transfer error rates")
    __plot_transfer_error_rates(args, torperf_dbs, tornet_dbs, "ALL")

    logging.info("Loading circuits info")
    tornet_dbs = __load_tornet_datasets(args, "circuit_list.json")
    logging.info("Plotting circuit num")
    __plot_client_circuits(args, tornet_dbs)

    print_current_memory("Before loading circuit data")
    #circuit_dict_db = __load_tornet_datasets(args, "circuit_dict.json")
    circuit_bandwidth_db = __load_tornet_datasets(args, "circuit_bandwidth.json")
    logging.info("Simulating attacker")
    print_current_memory("After loading circuit data")
    # Disable for now as it takes a very long time and yields no results
    #__plot_attacker(args, args.tornet_collection_path, tornet_dbs, circuit_dict_db, circuit_bandwidth_db)
    #__plot_entropy(args, args.tornet_collection_path, tornet_dbs, circuit_dict_db)
    __plot_circuit_bandwidth(args, circuit_bandwidth_db)
    args.pdfpages.close()

def get_relay_capacities(shadow_config_path, bwup=False, bwdown=False):
    relays = {}
    if not bwup and not bwdown:
        return relays
    if shadow_config_path is None or not pathlib.Path(shadow_config_path).exists():
        print("Failed to open shadow config...")
        return relays
    from lxml import etree
    # shadow_config_path should be a specific file
    # this will go through all the relays listed
    # and extract the "true" bandwidth for each
    # return a dict of nickname->true_bandwidth
    parser = etree.XMLParser(remove_blank_text=True)
    tree = etree.parse(shadow_config_path, parser)
    root = tree.getroot()
    for n in root.iterchildren("host"):
        nick = n.get('id')
        if 'relay' not in nick and 'thority' not in nick:
            continue
        l = []
        if bwup:
            if n.get('bandwidthup') != None:
                l.append(int(n.get('bandwidthup'))/1024.0) # KiB/s to MiB/s
            else:
                continue
        if bwdown:
            if n.get('bandwidthdown') != None:
                l.append(int(n.get('bandwidthdown'))/1024.0) # KiB/s to MiB/s
            else:
                continue
        relays[nick] = min(l)
    return relays

def select_bad_nodes(node_bw_dict, fraction = .1, sort_func = lambda l: random.sample(l, len(l)), min_bw = 0, max_bw = 2**64):
    total_bw = sum(node_bw_dict.values())
    target_bw = total_bw * fraction
    current_bw = 0
    available_nodes = list(node_bw_dict.keys())
    available_nodes = sort_func(available_nodes)
    nodes = []
    while current_bw < target_bw and len(available_nodes) > 0:
        node = available_nodes.pop(0)
        # Only add a new node if the resulting bandwidth is lower than the target. Prevents adding a very large bandwidth relay at the end
        if current_bw + node_bw_dict[node] < target_bw and node_bw_dict[node] >= min_bw and node_bw_dict[node] <= max_bw:
            current_bw += node_bw_dict[node]
            nodes.append(node)

    return nodes


def __get_current_circuits(curr_circuits, circ_change_list):
    for circ_change in circ_change_list:
        is_remove = False
        if circ_change[0] == "-":
            is_remove = True
            circ_change = circ_change[1:]
        #client_id, circuit_id = circ_change.split("_")
        #print("[{}] {}: {}. {}".format(time, client_id, circuit_id, is_remove))
        #print(circuit_list_db[experiment_id]["dataset"][0]["markovclient"].keys())
        if not is_remove:
            curr_circuits.append(circ_change)
        else:
            if circ_change in curr_circuits:
                curr_circuits.remove(circ_change)
    return curr_circuits


def __plot_entropy(args, tornet_collection_path, circuit_list, circuit_list_db):
    shadow_config_path = "{}/shadow.config.xml".format(tornet_collection_path[0])
    relay_list = get_relay_capacities(shadow_config_path, bwup=True, bwdown=True)
    guard_list = {k:v for (k,v) in relay_list.items() if "guard" in k}
    exit_list = {k:v for (k,v) in relay_list.items() if "exit" in k}
    all_combinations = []
    for guard_relay in guard_list:
        for exit_relay in exit_list:
            if exit_relay != guard_relay:
                all_combinations.append([guard_relay, exit_relay])
    for experiment_id in range(0, len(circuit_list_db)):
        circuit_list_db[experiment_id]["data"] = []
        circuit_dict = []
        norm_entropy = defaultdict(float)
        for time, circ_change_list in circuit_list[experiment_id]["dataset"][0]["markovclient"].items():
            circuit_dict = __get_current_circuits(circuit_dict, circ_change_list)

            used_circuits = []
            # calculate entropy over circuit_dict
            for open_circ in circuit_dict:
                used_relays = circuit_list_db[experiment_id]["dataset"][0]["markovclient"][open_circ]
                if len(used_relays) == 3:
                    if "guard" in used_relays[0] and "exit" in used_relays[2]:
                        used_circuits.append([used_relays[0], used_relays[2]])
            _, counts = np.unique(used_circuits, return_counts=True, axis=0)
            prob_list = np.divide(counts, len(used_circuits))
            normalized_entropy = entropy(prob_list, base=2)/math.log(len(all_combinations), 2)
            if normalized_entropy > 1:
                print("##### Entropy over 1")
                #print(f"{len(all_combinations)=} {entropy(prob_list)=} {sum(prob_list)=} {prob_list=} {counts=} {len(used_circuits)=} {all_combinations=} {used_circuits=}")
            norm_entropy[int(time)] = [normalized_entropy]
        circuit_list_db[experiment_id]["data"].append(list(norm_entropy.values()))
        circuit_list[experiment_id]["data"] = norm_entropy
    __plot_timeseries_figure(args, circuit_list, "Entropy", xtime=True, xlabel="Simulated Time (Minutes)", ylabel="Normalized Entropy")
    __plot_cdf_figure(args, circuit_list_db, 'entropy_cdf', xlabel="Entropy")


def __plot_circuit_bandwidth(args, circuit_bandwidth_db):
    dbs_to_plot = []
    for experiment_id in range(0, len(circuit_bandwidth_db)):
        for dataset in circuit_bandwidth_db[experiment_id]["dataset"]:
            circuit_bw_list = []
            for node, val in dataset.items():
                for circ_id, bw_time_dict in val.items():
                    circuit_bw = 0
                    for time, bw_dict in bw_time_dict.items():
                        circuit_bw += bw_dict["WRITTEN"]
                        circuit_bw += bw_dict["READ"]
                    circuit_bw_list.append(circuit_bw/float(1e6))
        circuit_bandwidth_db[experiment_id]["data"] = [circuit_bw_list]
        dbs_to_plot.append(circuit_bandwidth_db[experiment_id])
    __plot_cdf_figure(args, dbs_to_plot, "bytes_per_circuit", xlabel="Megabyte per circuit")




def __plot_attacker(args, tornet_collection_path, circuit_list_db, circuit_dict_db, circuit_bandwidth_db):
    if len(set(map(len,[circuit_list_db, circuit_dict_db, circuit_bandwidth_db]))) != 1:
        print("Databases have different lengths. Can't plot attacker!")
        return

    shadow_config_path = "{}/shadow.config.xml".format(tornet_collection_path[0])
    relay_list = get_relay_capacities(shadow_config_path, bwup=True, bwdown=True)
    guard_list = {k:v for (k,v) in relay_list.items() if "guard" in k}
    exit_list = {k:v for (k,v) in relay_list.items() if "exit" in k}

    # determine bad nodes
    bad_guards_dict = defaultdict(dict)
    bad_exits_dict = defaultdict(dict)

    for i in range(0, 10):
        bad_guards_dict["random"][i] = select_bad_nodes(guard_list)
        bad_exits_dict["random"][i] = select_bad_nodes(exit_list)
        print("Selected {} bad guards and {} bad exits in iteration {}".format(len(bad_guards_dict["random"][i]), len(bad_exits_dict["random"][i]), i))

    # iterate through all circuits and count traffic through bad nodes
    # [run][num,written,read, connections]
    bad_stats = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    bad_types = ["circuit", "guard", "exit"]
    direction_types = ["write", "read", "count"]
    print_current_memory("Before main attacmer loop")
    for experiment_id in range(0, len(circuit_list_db)):
        for bad_name in bad_guards_dict.keys():
            for i in bad_guards_dict[bad_name].keys():
                for t in bad_types:
                    bad_stats[experiment_id][t][bad_name][i] = [0,defaultdict(int),defaultdict(int), defaultdict(int)]
                current_bad_list = bad_guards_dict[bad_name][i]

                print_current_memory("Before iterating through nodes")
                # [node] = list of cids
                for node, val in circuit_bandwidth_db[experiment_id]["dataset"][0].items():
                    # [name] = list of cids
                    bad_cids = defaultdict(list)
                    # ignore internal circuits
                    if "markovclient" in node:
                        for circuit_id, circuit_dict in val.items():
                            #print(f"{circuit_id} : list: {circuit_dict}\n")
                            try:
                                circuit_nodes = circuit_dict_db[experiment_id]["dataset"][0][node][circuit_id]
                                if len(circuit_nodes) != 3:
                                    # skip one hops etc
                                    continue
                                node_names = [x.split("~")[1] for x in circuit_nodes]
                                bad_traffic_types = []
                                if node_names[0] in bad_guards_dict[bad_name][i]:
                                    bad_traffic_types.append("guard")
                                if node_names[2] in bad_guards_dict[bad_name][i]:
                                    bad_traffic_types.append("exit")
                                if len(bad_traffic_types) == 2:
                                    bad_traffic_types.append("circuit")
                                for t, types in circuit_bandwidth_db[0]["dataset"][0]["markovclient"][circuit_id].items():
                                    read_bytes = types["DELIVERED_READ"]
                                    written_bytes = types["DELIVERED_WRITTEN"]
                                    for bad_traffic_type in bad_traffic_types:
                                        bad_stats[experiment_id][bad_traffic_type][bad_name][i][0] +=1
                                        bad_stats[experiment_id][bad_traffic_type][bad_name][i][1][t] += written_bytes
                                        bad_stats[experiment_id][bad_traffic_type][bad_name][i][2][t] += read_bytes
                            except:
                                #print("Error locating {}".format(circuit_id))
                                #print(traceback.format_exc())
                                #print(circuit_dict_db[experiment_id]["dataset"][0][node].keys())
                                continue


        print_current_memory("After collecting data")
        logging.info("Finished collecting attacker bandwidth data. Plotting now...")
    for bad_traffic_type in bad_types:
        for bad_traffic_selection_name in bad_guards_dict.keys():
            dbs_to_plot = defaultdict(list)
            for experiment_id in range(0, len(circuit_list_db)):
                print_current_memory("Plot {} {} with id {}".format(bad_traffic_type, bad_traffic_selection_name, experiment_id))
                counts = []
                written = []
                read = []
                bad_connections = []
                avg_written = defaultdict(list)
                avg_read = defaultdict(list)
                avg_bad_connections = defaultdict(list)
                for e, val in bad_stats[experiment_id][bad_traffic_type][bad_traffic_selection_name].items():
                    counts.append(val[0])
                    written.append(sum(map(int, val[1].values())))
                    read.append(sum(map(int, val[2].values())))
                    bad_connections.append(sum(val[3].values()))
                    for t, throughput in val[1].items():
                        avg_written[int(t)] = [sum([throughput] + avg_written[int(t)])]
                    for t, throughput in val[2].items():
                        avg_read[int(t)] = [sum([throughput] + avg_read[int(t)])]
                    for t, bad in val[3].items():
                        avg_bad_connections[int(t)] = [sum([bad] + avg_bad_connections[int(t)])]

                db_copy = copy.deepcopy(circuit_bandwidth_db[experiment_id])
                db_copy["data"] = avg_written
                #print("Added written with len {}. Len of bad_stats {}".format(len(avg_written.keys()), bad_stats[experiment_id][bad_traffic_type][bad_traffic_selection_name].keys()))
                #print("bad_stats[{}] t {} e {} keys: {}".format(experiment_id, t, e, bad_stats[experiment_id].keys()))
                dbs_to_plot["write"].append(db_copy)
                db_copy = copy.deepcopy(circuit_bandwidth_db[experiment_id])
                db_copy["data"] = avg_read
                dbs_to_plot["read"].append(db_copy)

            dict_keys = list(dbs_to_plot.keys())
            for d in dict_keys:
                print("e {} d {} t {}".format(bad_traffic_selection_name, d, bad_traffic_type))
                __plot_timeseries_figure(args, dbs_to_plot[d], "{} attacker selection for {} ({})".format(bad_traffic_selection_name, bad_traffic_type, d), xtime=True, xlabel="Simulated Time (Minutes)", ylabel="Comprimised bytes ({} {} {})".format(bad_traffic_selection_name, d, bad_traffic_type))
                del dbs_to_plot[d]
    print("Finished selecting bad nodes")

def __plot_node_memory_usage(args, tornet_dbs):
    print("#################")
    node_types = ["client", "guard", "exit", "middle", "4uthority"]
    dbs_to_plot = {}
    for t in node_types:
        dbs_to_plot[t] = []
    for tornet_db in tornet_dbs:
        xy = {}
        # [name][second] = [ram vals]
        node_ram_sum = defaultdict(lambda: defaultdict(lambda: [0,0,0,0,0,0]))
        for i, d in enumerate(tornet_db['dataset']):
            for node_name, node_times in d["node_ram"].items():

                for node_time, ram_vals in node_times.items():
                    ram_vals = list(map(int, ram_vals))
                    alloc_bytes = ram_vals[1]
                    dealloc_bytes = ram_vals[2]
                    for t in node_types:
                        if t in node_name:
                            node_ram_sum[t][node_time] = list(map(operator.add, node_ram_sum[t][node_time], ram_vals))
        for t, val in node_ram_sum.items():
            plot_data = {}
            for time, ram_vals in val.items():
                time = float(time)
                total_bytes = ram_vals[3]
                plot_data[time] = [total_bytes/1073742000.0]
            db_copy = copy.deepcopy(tornet_db)
            db_copy["data"] = plot_data
            dbs_to_plot[t].append(db_copy)

    for t in node_types:
        __plot_timeseries_figure(args, dbs_to_plot[t], "ram_{}".format(t), xtime=True, xlabel="Simulated Time (Minutes)", ylabel="RAM Used (GiB) for {}".format(t))

def __plot_node_cpu_usage(args, tornet_dbs):
    node_types = ["client", "guard", "exit", "middle", "4uthority"]
    dbs_to_plot = {}
    for t in node_types:
        dbs_to_plot[t] = []
    for tornet_db in tornet_dbs:
        xy = {}
        # [name][second] = cpu time
        node_cpu = defaultdict(lambda: defaultdict(int))
        for i, d in enumerate(tornet_db['dataset']):
            for node_name, node_times in d["node_cpu"].items():

                for node_time, cpu_val in node_times.items():
                    for t in node_types:
                        if t in node_name:
                            node_cpu[t][node_time] = int(cpu_val)
        for t, val in node_cpu.items():
            plot_data = {}
            for time, cpu_vals in val.items():
                time = float(time)
                plot_data[time] = [cpu_vals]
            db_copy = copy.deepcopy(tornet_db)
            db_copy["data"] = plot_data
            dbs_to_plot[t].append(db_copy)

    for t in node_types:
        __plot_timeseries_figure(args, dbs_to_plot[t], "cpu_{}".format(t), xtime=True, xlabel="Simulated Time (Minutes)", ylabel="CPU Time (TODO) for {}".format(t))

def __plot_memory_usage(args, tornet_dbs):
    for tornet_db in tornet_dbs:
        xy = {}
        for i, d in enumerate(tornet_db['dataset']):
            if 'ram' not in d or 'gib_used_per_minute' not in d['ram']:
                continue
            timed = d['ram']['gib_used_per_minute']
            for sim_minute in timed:
                s = int(sim_minute)*60.0 # to seconds
                xy.setdefault(s, []).append(timed[sim_minute])
        tornet_db['data'] = xy

    dbs_to_plot = tornet_dbs

    __plot_timeseries_figure(args, dbs_to_plot, "ram",
        xtime=True,
        xlabel="Real Time",
        ylabel="RAM Used (GiB)")

def __plot_run_time(args, tornet_dbs):
    for tornet_db in tornet_dbs:
        xy = {}
        for i, d in enumerate(tornet_db['dataset']):
            if 'run_time' not in d or 'real_seconds_per_sim_second' not in d['run_time']:
                continue
            timed = d['run_time']['real_seconds_per_sim_second']
            for sim_secs in timed:
                s = int(round(float(sim_secs)))
                xy.setdefault(s, []).append(timed[sim_secs])
        tornet_db['data'] = xy

    dbs_to_plot = tornet_dbs

    __plot_timeseries_figure(args, dbs_to_plot, "run_time",
        ytime=True, xtime=True,
        xlabel="Simulation Time",
        ylabel="Real Time")

def __plot_relay_goodput(args, torperf_dbs, tornet_dbs, net_scale):
    # cache the corresponding data in the 'data' keyword for __plot_cdf_figure
    for tornet_db in tornet_dbs:
        tornet_db['data'] = []
        for i, d in enumerate(tornet_db['dataset']):
            l = [b/(1024**3)*8 for b in d.values()] # bytes to gbits
            tornet_db['data'].append(l)
    for torperf_db in torperf_dbs:
        gput = torperf_db['dataset']['relay_goodput']
        torperf_db['data'] = [[net_scale*gbits for gbits in gput.values()]]

    dbs_to_plot = torperf_dbs + tornet_dbs

    __plot_cdf_figure(args, dbs_to_plot, 'relay_goodput',
        xlabel="Sum of Relays' Goodput (Gbit/s)")

    for tornet_db in tornet_dbs:
        xy = {}
        for i, d in enumerate(tornet_db['dataset']):
            for key, val in d.items():
                l = val * 1e-6 # bytes to gbits
                xy.setdefault(int(key), []).append(l)
        tornet_db['data'] = xy

    dbs_to_plot = tornet_dbs
    __plot_timeseries_figure(args, dbs_to_plot, "relay_goodput",
        ytime=False, xtime=True,
        xlabel="Simulation Time",
        ylabel="Throughput (MiB/s)")

def __plot_circuit_build_time(args, torperf_dbs, tornet_dbs):
    # cache the corresponding data in the 'data' keyword for __plot_cdf_figure
    for tornet_db in tornet_dbs:
        tornet_db['data'] = tornet_db['dataset']
    for torperf_db in torperf_dbs:
        torperf_db['data'] = [torperf_db['dataset']['circuit_build_times']]

    dbs_to_plot = torperf_dbs + tornet_dbs

    __plot_cdf_figure(args, dbs_to_plot, 'circuit_build_time',
        yscale="taillog",
        xlabel="Circuit Build Time (s)")

def __plot_round_trip_time(args, torperf_dbs, tornet_dbs):
    # cache the corresponding data in the 'data' keyword for __plot_cdf_figure
    for tornet_db in tornet_dbs:
        tornet_db['data'] = tornet_db['dataset']
    for torperf_db in torperf_dbs:
        torperf_db['data'] = [torperf_db['dataset']['circuit_rtt']]

    dbs_to_plot = torperf_dbs + tornet_dbs

    __plot_cdf_figure(args, dbs_to_plot, 'round_trip_time',
        yscale="taillog",
        xlabel="Circuit Round Trip Time (s)")

def __plot_transfer_time(args, torperf_dbs, tornet_dbs, bytes_key):
    # cache the corresponding data in the 'data' keyword for __plot_cdf_figure
    for tornet_db in tornet_dbs:
        tornet_db['data'] = [tornet_db['dataset'][i][bytes_key] for i, _ in enumerate(tornet_db['dataset']) if bytes_key in tornet_db['dataset'][i]]
    for torperf_db in torperf_dbs:
        # Older datasets don't have download data
        try:
            torperf_db['data'] = [torperf_db['dataset']['download_times'][bytes_key]]
        except:
            pass

    dbs_to_plot = torperf_dbs + tornet_dbs

    __plot_cdf_figure(args, dbs_to_plot, f"transfer_time_{bytes_key}",
        yscale="taillog",
        xlabel=f"Transfer Time (s): Bytes={bytes_key}")

def __plot_transfer_error_rates(args, torperf_dbs, tornet_dbs, error_key):
    # cache the corresponding data in the 'data' keyword for __plot_cdf_figure
    for tornet_db in tornet_dbs:
        tornet_db['data'] = [tornet_db['dataset'][i][error_key] for i, _ in enumerate(tornet_db['dataset']) if error_key in tornet_db['dataset'][i]]
    for torperf_db in torperf_dbs:
        err_rates = __compute_torperf_error_rates(torperf_db['dataset']['daily_counts'])
        torperf_db['data'] = [err_rates]

    dbs_to_plot = torperf_dbs + tornet_dbs

    __plot_cdf_figure(args, dbs_to_plot, f"transfer_error_rates_{error_key}",
        xlabel=f"Transfer Error Rate (\%): Type={error_key}")

def __plot_client_goodput(args, torperf_dbs, tornet_dbs):
    # Tor computes goodput based on the time between the .5 MiB byte to the 1 MiB
    # byte in order to cut out circuit build and other startup costs.
    # https://metrics.torproject.org/reproducible-metrics.html#performance

    # cache the corresponding data in the 'data' keyword for __plot_cdf_figure
    for tornet_db in tornet_dbs:
        tornet_db['data'] = tornet_db['dataset']
    for torperf_db in torperf_dbs:
        # convert tor's microseconds into seconds
        client_gput = [t/1000000.0 for t in torperf_db['dataset']["client_goodput"]]
        torperf_db['data'] = [client_gput]

    dbs_to_plot = torperf_dbs + tornet_dbs

    __plot_cdf_figure(args, dbs_to_plot, 'client_goodput',
        yscale="taillog",
        xlabel="Client Transfer Goodput (Mbit/s): 0.5 to 1 MiB")


def __plot_client_circuits(args, tornet_dbs):
    total_circuit_dbs = []
    circuit_time_dbs = []
    for tornet_db in tornet_dbs:
        for dataset in tornet_db["dataset"]:
            circuit_num = {}
            total_circuit_num_dict = {}
            total_circuit_num = 0
            current_circuits = set()
            # for plotting circuit time cdf
            circuit_time_dict = {}
            circuit_time_list = []
            for time, circ_list in dataset["markovclient"].items():
                for circ in circ_list:
                    if circ[0] == "-":
                        try:
                            current_circuits.remove(circ[1:])
                            circuit_time_list.append(float(time) - circuit_time_dict[circ[1:]])
                        except KeyError:
                            # Removing circuit before starting our data capture
                            pass
                    else:
                        if circ not in current_circuits:
                            total_circuit_num += 1
                            current_circuits.add(circ)
                            circuit_time_dict[circ] = float(time)

                circuit_num[int(time)] = [len(current_circuits)/1000.0]
                total_circuit_num_dict[int(time)] = [total_circuit_num / float(1e6)]

            db_copy = copy.deepcopy(tornet_db)
            db_copy["data"] = total_circuit_num_dict
            total_circuit_dbs.append(db_copy)
            # circuit times
            db_copy = copy.deepcopy(tornet_db)
            db_copy["data"] = [circuit_time_list]
            circuit_time_dbs.append(db_copy)


            tornet_db['data'] = circuit_num

    dbs_to_plot = tornet_dbs

    __plot_timeseries_figure(args, dbs_to_plot, "used_circuits",
        ytime=False, xtime=True,
        xlabel="Simulation Time",
        ylabel="Active Circuits (thousand)")

    __plot_timeseries_figure(args, total_circuit_dbs, "total_circuits",
        ytime=False, xtime=True,
        xlabel="Simulation Time",
        ylabel="Total created Circuits (million)")

    __plot_cdf_figure(args, circuit_time_dbs, "circuit_times", xlabel="Circuit times")

def __plot_cdf_figure(args, dbs, filename, xscale=None, yscale=None, xlabel=None, ylabel="CDF"):
    color_cycle = cycle(DEFAULT_COLORS)
    linestyle_cycle = cycle(DEFAULT_LINESTYLES)

    f = pyplot.figure()
    lines, labels = [], []

    for db in dbs:
        if 'data' not in db or len(db['data']) < 1:
            continue
        elif len(db['data']) == 1:
            plot_func, d = draw_cdf, db['data'][0]
        else:
            plot_func, d = draw_cdf_ci, db['data']

        if len(d) < 1:
            continue

        line = plot_func(pyplot, d,
            label=db['label'],
            color=db['color'] or next(color_cycle),
            linestyle=next(linestyle_cycle))

        lines.append(line)
        labels.append(db['label'])

    if xscale is not None:
        pyplot.xscale(xscale)
        if xlabel != None:
            xlabel += __get_scale_suffix(xscale)
    if yscale != None:
        pyplot.yscale(yscale)
        if ylabel != None:
            ylabel += __get_scale_suffix(yscale)
    if xlabel != None:
        pyplot.xlabel(xlabel)
    if ylabel != None:
        pyplot.ylabel(ylabel)

    m = 0.025
    pyplot.margins(m)

    # the plot will exit the visible space at the 99th percentile,
    # so make sure the x-axis is centered correctly
    # (this is usually only a problem if using the 'taillog' yscale)
    x_visible_max = None
    for db in dbs:
        if len(db['data']) >= 1 and len(db['data'][0]) >= 1:
            q = quantile(db['data'][0], 0.99)
            x_visible_max = q if x_visible_max == None else max(x_visible_max, q)
    if x_visible_max != None:
        pyplot.xlim(xmin=-m*x_visible_max, xmax=(m+1)*x_visible_max)

    __plot_finish(args, lines, labels, filename)

def __plot_timeseries_figure(args, dbs, filename, xtime=False, ytime=False, xlabel=None, ylabel=None):
    color_cycle = cycle(DEFAULT_COLORS)
    linestyle_cycle = cycle(DEFAULT_LINESTYLES)

    f = pyplot.figure()
    lines, labels = [], []

    for db in dbs:
        if 'data' not in db or len(db['data']) < 1:
            continue

        x = sorted(db['data'].keys())
        y_buckets = [db['data'][k] for k in x]

        if len(db['dataset']) > 1:
            plot_func = draw_line_ci
        else:
            plot_func = draw_line

        line = plot_func(pyplot, x, y_buckets,
            label=db['label'],
            color=db['color'] or next(color_cycle),
            linestyle=next(linestyle_cycle))

        lines.append(line)
        labels.append(db['label'])

    if xlabel != None:
        pyplot.xlabel(xlabel)
    if ylabel != None:
        pyplot.ylabel(ylabel)

    if xtime:
        f.axes[0].xaxis.set_major_formatter(FuncFormatter(__time_format_func_minutes))
        # this locates y-ticks at the hours
        f.axes[0].xaxis.set_major_locator(MultipleLocator(base=600))
        # rotate xlabels so they don't overlap
        pyplot.xticks(rotation=30)
    if ytime:
        f.axes[0].yaxis.set_major_formatter(FuncFormatter(__time_format_func))

    __plot_finish(args, lines, labels, filename)

def __plot_finish(args, lines, labels, filename):
    pyplot.tick_params(axis='both', which='major', labelsize=8)
    pyplot.tick_params(axis='both', which='minor', labelsize=5)
    pyplot.grid(True, axis='both', which='minor', color='0.1', linestyle=':', linewidth='0.5')
    pyplot.grid(True, axis='both', which='major', color='0.1', linestyle=':', linewidth='1.0')

    pyplot.legend(lines, labels, loc='best')
    pyplot.tight_layout(pad=0.3)
    pyplot.savefig(f"{args.prefix}/{filename}.{'png' if args.plot_pngs else 'pdf'}")
    args.pdfpages.savefig()

def __get_scale_suffix(scale):
    if scale == 'taillog':
        return " (tail log scale)"
    elif scale == 'log':
        return " (log scale)"
    else:
        return ""

def __time_format_func_minutes(x, pos):
    return "{:d}".format(int(x/60))


def __time_format_func(x, pos):
    hours = int(x//3600)
    minutes = int((x%3600)//60)
    seconds = int(x%60)
    return "{:d}:{:02d}:{:02d}".format(hours, minutes, seconds)

def __load_tornet_datasets(args, filename):
    tornet_dbs = []

    print("label")
    print(args.labels)
    label_cycle = cycle(args.labels) if args.labels != None else None
    color_cycle = cycle(args.colors) if args.colors != None else None

    if args.tornet_collection_path != None:
        for collection_dir in args.tornet_collection_path:
            tornet_db = {
                'dataset': [load_json_data(p) for p in find_matching_files_in_dir(collection_dir, filename)],
                'label': next(label_cycle) if label_cycle != None else os.path.basename(collection_dir),
                'color': next(color_cycle) if color_cycle != None else None,
            }
            tornet_dbs.append(tornet_db)

    return tornet_dbs

def __load_torperf_datasets(torperf_argset):
    torperf_dbs = []

    if torperf_argset != None:
        for torperf_args in torperf_argset:
            torperf_db = {
                'dataset': load_json_data(torperf_args[0]) if torperf_args[0] != None else None,
                'label': torperf_args[1] if torperf_args[1] != None else "Public Tor",
                'color': torperf_args[2],
            }
            torperf_dbs.append(torperf_db)

    return torperf_dbs

def __get_simulated_network_scale(args):
    sim_info = __load_tornet_datasets(args, "simulation_info.json")

    net_scale = 0.0
    for db in sim_info:
        for i, d in enumerate(db['dataset']):
            if 'net_scale' in d:
                if net_scale == 0.0:
                    net_scale = float(d['net_scale'])
                    logging.info(f"Found simulated network scale {net_scale}")
                else:
                    if float(d['net_scale']) != net_scale:
                        logging.warning("Some of your tornet data is from networks of different scale")
                        logging.critical(f"Found network scales {net_scale} and {float(d['net_scale'])} and they don't match")

    return net_scale

def __compute_torperf_error_rates(daily_counts):
    err_rates = []
    for day in daily_counts:
        year = int(day.split('-')[0])
        month = int(day.split('-')[1])

        total = int(daily_counts[day]['requests'])
        if total <= 0:
            continue

        timeouts = int(daily_counts[day]['timeouts'])
        failures = int(daily_counts[day]['failures'])

        err_rates.append((timeouts+failures)/float(total)*100.0)
    return err_rates
