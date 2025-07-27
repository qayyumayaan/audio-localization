import random, math

import numpy as np

class Mic:
    def __init__(self, pos, freq, offset_frac=0):
        self.pos = pos
        self.freq = freq
        self.offset = offset_frac / freq
    
    def getTimeReached(self, wave):
        t = self.offset
        while t * Wave.speed < math.sqrt((self.pos[0] - wave.pos[0])**2 + (self.pos[1] - wave.pos[1])**2):
            t += 1 / self.freq
        return t
    
    def calcTrueTime(self, wave):
        return math.sqrt((self.pos[0] - wave.pos[0])**2 + (self.pos[1] - wave.pos[1])**2) / Wave.speed

class Environment:
    temp = 68

    def __init__(self, mics, wave):
        self.mics = mics
        self.wave = wave
    
    def getMics(self):
        return self.mics
    
    def getWave(self):
        return self.wave
    
class Wave:
    speed = 331 + 0.6 * (5/9 * (Environment.temp - 32))

    def __init__(self, pos):
        self.pos = pos


def getRandomEnv(mics, maxRad):
    center_x = mics[0].pos[0]
    center_y = mics[0].pos[1]

    r = maxRad * math.sqrt(random.random())
    theta = 2 * math.pi * random.random()
    wave = Wave((center_x + r * math.cos(theta), center_y + r * math.sin(theta)))
    return Environment(mics, wave)

def getEstTDOA(base_mic, other_mic, wave):
    return other_mic.getTimeReached(wave) - base_mic.getTimeReached(wave)

def getTrueTDOA(base_mic, other_mic, wave):
    return other_mic.calcTrueTime(wave) - base_mic.calcTrueTime(wave)

if __name__ == "__main__":

    diffs = []
    for a, offset1 in enumerate([i / 100 for i in range(100)]):
        for b, offset2 in enumerate([j / 100 for j in range(100)]):
            mics = [Mic((0.0, 0.0), 192000), Mic((0.05, 0.0, offset1), 48000), Mic((0.025, 0.0433, offset2), 48000)]

            l = []
            env = getRandomEnv(mics, 100)
            for i, mic in enumerate(env.getMics()):
                if not mic.pos == mics[0].pos:
                    l.append(abs(getTrueTDOA(mics[0], mic, env.getWave()) - getEstTDOA(mics[0], mic, env.getWave())))

            diffs[a][b].append(np.mean(l))

    