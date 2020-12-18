from threading import Thread
from src.modules.supervisors.lane_supervisor import LaneSupervisor
from src.utils.templates.workerprocess import WorkerProcess


class SystemSupervisor(WorkerProcess):

    def __init__(self, inPs, outPs, pi_camera=None):
        super(SystemSupervisor, self).__init__(inPs, outPs)
        self._lane_supervisor = LaneSupervisor(pi_camera=pi_camera)
        print("System supervisor initialized")

    def run(self):
        print("Run system process")
        super(SystemSupervisor, self).run()

    def read_command(self, outPs):
        print("Trying to read command")
        command = self._lane_supervisor.get_command_dictionary()
        print(command)
        for outP in outPs:
            outP.send(command)

    def _init_threads(self):
        print("Initializing System threads")
        readTh = Thread(name="CommandReader", target=self.read_command, args=(self.outPs,))
        self.threads.append(readTh)
        pass
