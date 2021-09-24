import resource


def print_current_memory(s = ""):
    kbyte_mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    gbyte_mem = kbyte_mem * 1e-6
    print("[{}]Current max rss is {} Gigabyte".format(s, gbyte_mem))
