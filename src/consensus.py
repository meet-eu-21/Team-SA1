def consensus(list1, list2, resolution):
    resolution_to_gap = {100000: 2,
                         50000: 2,
                         25000: 4}
    gap = resolution_to_gap[resolution]
    true_TADs = []
    for tad in list1:
        for i in range(-gap, gap+1):
            for j in range(-gap, gap+1):
                if (tad[0]+i, tad[1]+j) in list2:
                    #print(tad, (tad[0]+i, tad[1]+j))
                    #print((min(tad[0], tad[0]+i), max(tad[1], tad[1]+j)))
                    true_TADs.append((min(tad[0], tad[0]+i), max(tad[1], tad[1]+j)))
    return true_TADs