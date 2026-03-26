
""" 
Muhammed Sait Çağlar
Department of Computer Engineering
150230009

"""
import cv2
import numpy as np
import math
import matplotlib.pyplot as plt


def TAD(frames): 
    TADs = []
    for i in range(len(frames) - 1):
        difference = np.abs(frames[i].astype(np.int32) - frames[i+1].astype(np.int32))
        tadValue = np.sum(difference)
        TADs.append(tadValue)
    return TADs


def findCandidates(frames, darkFrames):
    plt.figure(figsize=(12, 5))
    plt.plot(frames)
    plt.title("TAD vs Index")
    plt.xlabel("Index of Frame")
    plt.ylabel("TAD")
    plt.grid()
    plt.show()
    plt.savefig("TADGraph.png")
    meanVal = np.mean(frames)
    stdVal = np.std(frames)
    threshold = meanVal + 3 * stdVal # this method was used to determine TAD threshold. Best results were obtained with constant 3.
    candidates = []
    candidateIndex = [] #index of candidates
    for i in range(len(frames)):
        if frames[i] > threshold:
            candidates.append(darkFrames[i])
            candidateIndex.append(i)
    print(candidateIndex)

    return candidates,candidateIndex
        

def SVDforSingularvalues(A): # it is simplified svd calculation. Since efficiency issues , it was used. However, full form can be found ond appendix.
    toFindMatrix = A @ A.T
    eigenvalues = np.linalg.eigvals(toFindMatrix)
    sortedEigenvalues = np.sort(eigenvalues)[::-1] # to descending order
    E = np.zeros((len(sortedEigenvalues),1))
    for i in range(len(sortedEigenvalues)):
        E[i][0] = math.sqrt(max(sortedEigenvalues[i], 0))
    return E.flatten()

def findTransitions(svdCandidates,frames,candidateIndex):
    thresholdValues = 3.8 #threshold for video 1 and video2. it was found experimentally
    transitionCount = 0
    with open("C:/Users/Sait/Desktop/hw/transitions.txt", "w") as f:
        for i in range(len(svdCandidates)):
            difference = np.linalg.norm(np.abs(svdCandidates[i] - SVDforSingularvalues(frames[candidateIndex[i] + 1])))
            if difference > thresholdValues:
                transitionCount += 1
                f.write(f"Transition {transitionCount}: Between Frame {candidateIndex[i]} and Frame {candidateIndex[i] + 1}\n")
                fileName = f"C:/Users/Sait/Desktop/hw/transitionFrame{i+1}.jpg"
                nextfilename = f"C:/Users/Sait/Desktop/hw/afterTransition{transitionCount}.jpg"
                cv2.imwrite(fileName, frames[candidateIndex[i]])
                cv2.imwrite(nextfilename, frames[candidateIndex[i] + 1])
                



darkFrames = []
capture = cv2.VideoCapture("enter_the_video_path")
reducedSizes  = (100,75)
while True:
    isRead, frame = capture.read()
    if not isRead:
        print("Total frame numbers : ", len(darkFrames))
        break
    new = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    newResized = cv2.resize(new, reducedSizes, interpolation=cv2.INTER_LINEAR)
    darkFrames.append(newResized)
capture.release()
frames = TAD(darkFrames) # to compute TADs and returns a list of difference between consecutive frames 
candidates, indexOfCandidates = findCandidates(frames,darkFrames) # to find possible transitions
svdCandidates = []
for i in range(len(candidates)):
    svdCandidates.append(SVDforSingularvalues(candidates[i])) # to apply SVD to the candidates

findTransitions(svdCandidates,darkFrames,indexOfCandidates) # to check whether the candidates are transitions

    




