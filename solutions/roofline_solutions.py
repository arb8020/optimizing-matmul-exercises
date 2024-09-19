def calculate_roofline(m, n, k, memory_bandwidth, computational_performance, data_type_size):
    
    operations = 2 * m * n * k  # calculate number of operations
    loads = m * k + k * n  # calculate number of loads 
    stores = m * n  # calculate number of stores
    
    data_transfer = (loads + stores) * data_type_size  # total amount of data to transfer
    
    computation_time = operations / (computational_performance * 1e12)  # time to perform operations
    
    memory_bandwidth_bytes = memory_bandwidth * 1e9  # convert to bytes/second
    data_transfer_time = data_transfer / memory_bandwidth_bytes  # time to transfer data
    
    # in the lower bound, we can imagine that GPUs are able to start processing data as soon as part of it loads in
    # so the latency is only bounded by either compute time or data transfer time - whichever is larger
    lower_bound_latency = max(computation_time, data_transfer_time)
    
    # calculate performance (operations/time)
    # make sure to convert back to gigaflops
    gflops = operations / (lower_bound_latency * 1e9)
    
    return {
        'operations': operations,
        'data_transfer': data_transfer,
        'computation_time': computation_time,
        'data_transfer_time': data_transfer_time,
        'lower_bound_latency': lower_bound_latency,
        'gflops': gflops
    }
