#!/usr/bin/env python

import glob
import os
import sys

from collections import defaultdict
from multiprocessing import Process
from threading import RLock
import Pyro4
import argparse
import math
import numpy as np
import random
import time
if sys.version_info.major == 2:
    from pathlib2 import Path
else:
    from pathlib import Path

class CrowdService():
    def __init__(self):
        
        self._local_intentions = []
        self._local_intentions_lock = RLock()

   
    @property
    def local_intentions(self):
        return self._local_intentions

    @local_intentions.setter
    def local_intentions(self, velocities):
        self._local_intentions = velocities

    def acquire_local_intentions(self):
        self._local_intentions_lock.acquire()

    def release_local_intentions(self):
        try:
            self._local_intentions_lock.release()
        except Exception as e:
            print(e)
            sys.stdout.flush()
   
class XXX():
    def __init__(self) -> None:
        self.crowd_service = CrowdService()

    def add_random_intentions(self, count):
        local_intentions = []
        for i in range(count):
            local_intentions.append((i, "hihi", np.random.rand(2)))


        self.crowd_service.acquire_local_intentions()
        self.crowd_service.local_intentions = local_intentions
        self.crowd_service.release_local_intentions()


    def get_attentions(self):
        self.crowd_service.acquire_local_intentions()

        local_intentions = self.crowd_service.local_intentions

        self.crowd_service.release_local_intentions()

        print(local_intentions)

if __name__ == "__main__":
    xxx = XXX()
    xxx.add_random_intentions(2)
    xxx.get_attentions()

    xxx.get_attentions()

    xxx.add_random_intentions(2)


    xxx.get_attentions()