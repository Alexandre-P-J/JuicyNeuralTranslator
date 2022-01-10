import json
import matplotlib.pyplot as plt

# state = "/home/pyro/Projects/TFG/Checkpoints/Transfer/opus-mt-transfer-ca-to-en/checkpoint-95000/trainer_state.json"

# with open(state, "r") as file:
#     hist = json.loads(file.read())["log_history"]
#     x = []
#     y = []
#     ex = []
#     ey = []
#     for p in hist:
#         if "loss" in p:
#             x.append(p["epoch"])
#             y.append(p["loss"])
#         else:
#             ex.append(p["epoch"])
#             ey.append(p["eval_loss"])
#     plt.xlabel("epoch")
#     plt.ylabel("loss")
#     plt.plot(x, y, label="train")
#     ex.insert(0, 0)
#     ex.append(5.0)
#     ey.append(1.51)
#     ey.insert(0, 1.9)
#     plt.plot(ex, ey, label="val")
#     plt.legend()
#     plt.show()



# state = "/home/pyro/Projects/TFG/Checkpoints/Finetune/en-es-SciELO/run4/checkpoint-46500/trainer_state.json"

# with open(state, "r") as file:
#     hist = json.loads(file.read())["log_history"]
#     x = []
#     y = []
#     ex = []
#     ey = []
#     for p in hist:
#         if "loss" in p:
#             x.append(p["epoch"])
#             y.append(p["loss"])
#         else:
#             ex.append(p["epoch"])
#             ey.append(p["eval_loss"])
#     plt.xlabel("epoch")
#     plt.ylabel("loss")
#     plt.plot(x, y, label="train")
#     ex.insert(0, 0)
#     ex.append(3.0)
#     ey.append(1.1)
#     ey.insert(0, 1.24)
#     plt.plot(ex, ey, label="val")
#     plt.legend()
#     plt.show()



def plot(path):
    with open(path, "r") as file:
        hist = json.loads(file.read())["log_history"]
        x = []
        y = []
        ex = []
        ey = []
        for p in hist:
            if "loss" in p:
                x.append(p["epoch"])
                y.append(p["loss"])
            else:
                ex.append(p["epoch"])
                ey.append(p["eval_loss"])
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.plot(x, y, label="train")
        plt.plot(ex, ey, label="val")
        plt.legend()
        #plt.show()


base = "/home/pyro/Projects/TFG/Experimental/opus-mt-finetuned-en-to-es"
import os
import numpy as np

while (True):
    folders = os.listdir(base)
    x = [int("".join([c for c in f if c.isdigit()])) for f in folders]
    folder = folders[np.argmax(x)]
    path = os.path.join(os.path.realpath(base), folder, "trainer_state.json")
    plot(path)
    plt.pause(60)
    plt.clf()
