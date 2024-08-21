def ranges(nplanets, num_threads):
    for_per_thread = nplanets // num_threads
    ranges_for = []
    for i in range(num_threads):
        if(i!=11):
            start = for_per_thread*i
            final = for_per_thread*(i+1)
        else:
            start = for_per_thread*i
            final = nplanets
        ranges_for.append((start, final))
    return ranges_for