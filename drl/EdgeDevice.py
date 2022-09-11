class EdgeDevice:
    '所有子任务的基类'

    def __init__(self, deviceId, cpuNum, waitTime):
        self.deviceId = deviceId
        self.cpuNum = cpuNum
        self.waitTime = waitTime

    def setWaitTime(self, waitTime):
        self.waitTime = waitTime

    def printInfo(self):
        print("deviceId:"+str(self.deviceId)+",cpuNum:"+
                              str(self.cpuNum)+",waitTime:"+str(self.waitTime))
