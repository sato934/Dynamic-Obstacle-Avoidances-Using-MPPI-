import numpy as np
from Pos_Model import Pos_Model
from Ang_Model import Ang_Model

def Sim_Model(State, Ctrl, P):

    pos_state = Pos_Model(State, Ctrl, P)
    ang_state = Ang_Model(State, Ctrl, P)
    next_state = np.vstack([pos_state, ang_state])
    return next_state
