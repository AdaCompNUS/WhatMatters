from multiprocessing import Process
import collections
import sys
import os
import subprocess

scripts_path = os.path.dirname(os.path.realpath(__file__))
src_path = os.path.dirname(scripts_path)
moped_server_path = os.path.join(src_path, 'moped')
sys.path.append(moped_server_path)

def print_flush(msg):
    print(msg)
    sys.stdout.flush()


class MopedSimulatorAccessories(Process):
    def __init__(self, cmd_args, config):
        Process.__init__(self)
        self.verbosity = config.verbosity

        Args = collections.namedtuple('args', 'host moped_pyro_port')

        # Spawn meshes.
        self.args = Args(
            host='127.0.0.1',
            moped_pyro_port = config.moped_pyro_port)

    def run(self):
        if self.verbosity > 0:
            print_flush("[moped_simulator.py] Running moped simulator to genrate future plans")

        moped_pyro4_server_script = os.path.join(moped_server_path, "moped_pyro4_server.py")

        cmd = [moped_pyro4_server_script, "--host", self.args.host, "--mopedpyroport", str(self.args.moped_pyro_port)]
        
        #subprocess.run(cmd)


