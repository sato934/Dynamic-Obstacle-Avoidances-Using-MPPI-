import os
import pickle
import Load_Settings

def FolderCheck(j, k):
    P = Load_Settings.Load_Settings(j)

    DirName = f"DataSet_MPC/K-{P['K']}/H-{P['Horizon']}"

    if not os.path.exists(DirName):
        os.makedirs(DirName)

    FolderNum = 0
    while True:
        FolderNum += 1
        FolderName = f"{DirName}/No.{FolderNum}"
        if not os.path.exists(FolderName):
            os.makedirs(FolderName)
            break

    # 設定を保存（pickleでバイナリ保存）
    with open(f"{FolderName}/Settings.pkl", "wb") as f:
        pickle.dump(P, f)

    return FolderName
