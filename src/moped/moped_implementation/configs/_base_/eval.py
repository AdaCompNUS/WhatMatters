evaluation = dict(
    # Test Evaluation to upload to competitions. We do not need to have any configurations because output is supposed to upload
    # and evaluate online in the competition platform
    test_eval = dict(
    ),
    val_eval = dict(
        miss_threshold = 2.0, # Threshold for miss rate
        prune_n_guesses = 0, #Pruned number of guesses of non-map baseline using map
        n_guesses_cl=0, #Number of guesses along each centerline
        n_cl=0, #Number of centerlines to consider
        max_neighbors_cl=3 #Number of neighbors obtained for each centerline by the baseline
        #max_n_guesses=0, No use here. Default would be [1,3,6]
    )
)