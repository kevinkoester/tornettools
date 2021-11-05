import os
import logging
import datetime
import re

from numpy import mean

from tornettools.util import open_readable_file, load_json_data, dump_json_data

from collections import defaultdict

def __get_file(prefix, name, extension = "json"):
    json_path = f"{prefix}/{name}.{extension}"

    if not os.path.exists(json_path):
        json_path += ".xz"

    if not os.path.exists(json_path):
        logging.warning(f"Unable to find {extension} file at {json_path}.")
        return None
    return json_path

def parse_cpu_node_logs(args):
    return __parse_shadow_cpu(args.prefix)

def parse_resource_usage_node_logs(args):
    return __parse_shadow_memory(args)

def parse_resource_usage_logs(args):
    logging.info("Parsing resource usage from free log")
    if __parse_free_rusage(args):
        logging.info("Parsing resource usage from shadow log")
        return __parse_shadow_rusage(args)
    else:
        return False

def __parse_free_rusage(args):
    free_filepath = __get_file(args.prefix, "free", "log")
    if free_filepath is None:
        return False

    rusage = {}

    last_ts = None
    mem_header = None
    with open_readable_file(free_filepath) as inf:
        try:
            for line in inf:
                if "UTC" in line:
                    parts = line.strip().split()
                    if len(parts) >= 1:
                        ts = float(parts[0])
                        #dt = datetime.datetime.fromtimestamp(ts)
                        #last_ts = dt.timestamp()
                        last_ts = ts
                elif 'total' in line and mem_header == None:
                    mem_header = [p.strip() for p in line.strip().split()]
                elif "Mem:" in line:
                    parts = [p.strip() for p in line.strip().split()]
                    mem_counts = [int(p) for p in parts[1:]]

                    memd = {f"mem_{mem_header[i]}": mem_counts[i] for i in range(len(mem_counts))}

                    if last_ts > 0:
                        rusage.setdefault(last_ts, memd)
        except:
            last_ts = -1

    if len(rusage) > 0:
        outpath = f"{args.prefix}/free_rusage.json.xz"
        dump_json_data(rusage, outpath, compress=True)
        return True
    else:
        logging.warning(f"Unable to parse memory usage data from {free_filepath}.")
        return False

def timestamp_to_seconds(stamp):
    parts = stamp.split(":")
    h, m, s = int(parts[0]), int(parts[1]), float(parts[2])
    seconds = h*3600.0 + m*60.0 + s
    return seconds

def __parse_shadow_memory(args):
    shadow_filepath = __get_file(args.prefix, "shadow", "log")
    if shadow_filepath is None:
        return False
    # [node][time] = [alloc, dealloc, total]
    memory_usage = defaultdict(dict)
    memory_re = re.compile("\[_tracker_logRAM\]")
    with open_readable_file(shadow_filepath) as inf:
        for line in inf:
            if memory_re.search(line) != None:
                parts = line.strip().split()
                node = parts[4][1:-1]
                sim_time = timestamp_to_seconds(parts[2])
                ram_parts = parts[9].split(",")
                memory_usage[node][sim_time] = ram_parts
    if len(memory_usage) > 0:
        outpath = f"{args.prefix}/shadow_memory.json.xz"
        dump_json_data(memory_usage, outpath, compress=True)
        return True
    else:
        logging.warning(f"Unable to parse resource usage data from {shadow_filepath}.")
        return False

def __parse_shadow_cpu(prefix):
    shadow_filepath = __get_file(prefix, "shadow", "log")
    if shadow_filepath is None:
        return False

    heartbeat = re.compile("\[shadow-heartbeat\] \[node\]")
    cpu_usage = defaultdict(lambda: defaultdict(int))
    with open_readable_file(shadow_filepath) as inf:
        for line in inf:
            if heartbeat.search(line) != None:
                parts = line.strip().split()
                if len(parts) >= 9:
                    sim_time = timestamp_to_seconds(parts[2])
                    node = parts[4]
                    payload = parts[9]
                    payload_parts = payload.split(";")
                    first_part_parts = payload_parts[0].split(",")
                    if len(first_part_parts) >= 8:
                        total_cpu = first_part_parts[6]
                        interval_cpu = first_part_parts[7]
                        cpu_usage[node][sim_time] = total_cpu

    if len(cpu_usage) > 0:
        outpath = f"{prefix}/shadow_cpu.json.xz"
        dump_json_data(cpu_usage, outpath, compress=True)
        return True
    else:
        logging.warning(f"Unable to parse cpu usage data from {shadow_filepath}.")
        return False



def __parse_shadow_rusage(args):
    shadow_filepath = __get_file(args.prefix, "shadow", "log")
    if shadow_filepath is None:
        return False

    rusage = {}
    heartbeat = re.compile("(_slave_heartbeat)|(_manager_heartbeat)")
    with open_readable_file(shadow_filepath) as inf:
        for line in inf:
            if heartbeat.search(line) != None:
                parts = line.strip().split()
                if len(parts) >= 13:
                    sim_time = float(parts[12]) # nanos e.g. 2000000000
                    std = datetime.timedelta(microseconds=sim_time/1000.0)
                    sim_secs = std.total_seconds()

                    if sim_secs not in rusage:
                        real_time = parts[0] # time e.g. 00:00:15.436056
                        rt_parts = real_time.split(':')
                        rtd = datetime.timedelta(hours=int(rt_parts[0]), minutes=int(rt_parts[1]), seconds=float(rt_parts[2]))

                        rund = {keyval.split('=')[0]: keyval.split('=')[1] for keyval in parts if '=' in keyval}
                        rund['real_time'] = rtd.total_seconds()

                        rusage[sim_secs] = rund

    if len(rusage) > 0:
        outpath = f"{args.prefix}/shadow_rusage.json.xz"
        dump_json_data(rusage, outpath, compress=True)
        return True
    else:
        logging.warning(f"Unable to parse resource usage data from {shadow_filepath}.")
        return False

def extract_cpu_node_plot_data(args):
    shadow_cpu_json_path = __get_file(args.prefix, "shadow_cpu", "json")
    shadow_cpu_data = load_json_data(shadow_cpu_json_path)

    outpath = f"{args.prefix}/tornet.plot.data/cpu_node.json"
    rusage = {"node_cpu": shadow_cpu_data}
    dump_json_data(rusage, outpath, compress=False)

def extract_resource_usage_node_plot_data(args):
    shadow_memory_json_path = __get_file(args.prefix, "shadow_memory", "json")
    shadow_memory_data = load_json_data(shadow_memory_json_path)

    outpath = f"{args.prefix}/tornet.plot.data/resource_usage_node.json"
    rusage = {"node_ram": shadow_memory_data}
    dump_json_data(rusage, outpath, compress=False)


def extract_resource_usage_plot_data(args):
    free_json_path = __get_file(args.prefix, "free_rusage", "json")
    shadow_json_path = __get_file(args.prefix, "shadow_rusage", "json")

    free_data = load_json_data(free_json_path)
    shadow_data = load_json_data(shadow_json_path)

    rusage = {"ram": __get_ram_usage(free_data), "run_time": __get_run_time(shadow_data)}
    outpath = f"{args.prefix}/tornet.plot.data/resource_usage.json"
    dump_json_data(rusage, outpath, compress=False)

def __get_ram_usage(data):
    used = {float(ts): data[ts]["mem_used"] for ts in data}

    ts_start = min(used.keys())
    mem_start = used[ts_start] # mem used by OS, i.e., before starting shadow
    mem_max = max(used.values())

    # subtract mem used by OS, get time offset from beginning of simulation
    gib_used_per_second = {int(ts-ts_start): (used[ts]-mem_start)/(1024.0**3) for ts in used}
    bytes_used_max = mem_max - mem_start
    gib_used_max = bytes_used_max/(1024.0**3)

    gib_minute_bins = {}
    for second in gib_used_per_second:
        gib_minute_bins.setdefault(int(second/60), []).append(gib_used_per_second[second])

    gib_used_per_minute = {minute: mean(gib_minute_bins[minute]) for minute in gib_minute_bins}

    return {"bytes_used_max": bytes_used_max, "gib_used_max": gib_used_max, "gib_used_per_minute": gib_used_per_minute}

def __get_run_time(data):
    real_seconds_per_sim_second = {float(sim_sec): float(data[sim_sec]["real_time"]) for sim_sec in data}
    times = list(real_seconds_per_sim_second.values())
    runtime = datetime.timedelta(seconds=max(times))

    return {"human": str(runtime),
        "seconds": runtime.total_seconds(),
        "minutes": runtime.total_seconds()/60.0,
        "hours": runtime.total_seconds()/3600.0,
        "real_seconds_per_sim_second": real_seconds_per_sim_second}
