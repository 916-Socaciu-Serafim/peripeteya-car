from threading import Thread
from src.modules.supervisors.lane_supervisor import LaneSupervisor
from src.utils.templates.workerprocess import WorkerProcess


class SystemSupervisor(WorkerProcess):

    def __init__(self, inPs, outPs):
        super(SystemSupervisor, self).__init__(inPs, outPs)
        print("System supervisor initialized")

    def run(self):
        print("Run system process")
        super(SystemSupervisor, self).run()

    def read_command_offset(self, inP, outPs):
        while True:
            command, offset = inP.recv()
            print("Sending command: ", command, "Offset: ", offset)
            for outP in outPs:
                outP.send(command)

    def _init_threads(self):
        print("Initializing System threads")
        lane_thread = Thread(name="ComandTransmitter", target=self.read_command_offset, args=(self.inPs[0], self.outPs,))
        self.threads.append(lane_thread)
        pass
