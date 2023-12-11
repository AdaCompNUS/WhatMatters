
PRINT_LOG = False

ORIGINAL_PUREPURSUIT_UPDATE_FREQUENCY_IN_TIME = 0.1 # 0.1 # Updating every 1 seconds -> Hz = 1 | 5.0 for 0.6 Hz, 0.02 ts,  if 10 times slower, use 1

ORIGINAL_CROWD_PROCESSOR_UPDATE_FREQUENCY_IN_HZ = 10 #10 # Updating every 1 seconds -> Hz = 1,  if 10 times slower, use 1

ORIGINAL_EGO_VEHICLE_UPDATE_FREQUENCY_IN_HZ = 20 #7, if 10 times slower, use 2

ORIGINAL_EGO_VEHICLE_PUBLISH_INFO_FREQUENCY_IN_TIME = 0.02 #0.02 # Updating every 0.5 seconds -> Hz = 2, if 10 times slower, use 0.2

# ORIGNAL

EGO_VEHICLE_UPDATE_FREQUENCY_IN_HZ = 20 #20, if 10 times slower, use 2


## 3 times slower (CV/CA)

# EGO_VEHICLE_UPDATE_FREQUENCY_IN_HZ = 6.667 #20, if 10 times slower, use 2



## 10 times slower (assume same  computation)

# EGO_VEHICLE_UPDATE_FREQUENCY_IN_HZ = 2 #20, if 10 times slower, use 2



## 15 times slower

# EGO_VEHICLE_UPDATE_FREQUENCY_IN_HZ = 1.33 #20, if 10 times slower, use 2



## 30 times slower (lanegcn/hivt)

#EGO_VEHICLE_UPDATE_FREQUENCY_IN_HZ = 0.667 #20, if 10 times slower, use 2



## 300 times slower (same Hz)

# I believe PUREPURSUIT_UPDATE_FREQUENCY_IN_TIME, EGO_PUBLISH and CROWD_UPDATE no need to change frequently.

#EGO_VEHICLE_UPDATE_FREQUENCY_IN_HZ = 0.0667 #20, if 10 times slower, use 2



## 600 times slower (same Hz)

#EGO_VEHICLE_UPDATE_FREQUENCY_IN_HZ = 0.0333 #20, if 10 times slower, use 2

