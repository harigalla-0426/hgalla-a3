#!/usr/bin/python
#
# Perform optical character recognition, usage:
#     python3 ./image2text.py train-image-file.png train-text.txt test-image-file.png
#
# Authors: hgalla-rathlur-vchitti
# (based on skeleton code by D. Crandall, Oct 2020)
#

from PIL import Image, ImageDraw, ImageFont
import sys

CHARACTER_WIDTH = 14
CHARACTER_HEIGHT = 25
TRAIN_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789(),.-!?\"' "


def load_letters(fname):
    im = Image.open(fname)
    px = im.load()
    # for x in range(CHARACTER_HEIGHT):
    #     for y in range(CHARACTER_WIDTH):
    #         print("px",px[x,y])
    (x_size, y_size) = im.size
    # print(im.size)
    # print(int(x_size / CHARACTER_WIDTH) * CHARACTER_WIDTH)
    result = []
    for x_beg in range(0, int(x_size / CHARACTER_WIDTH) * CHARACTER_WIDTH, CHARACTER_WIDTH):
        result += [["".join(['*' if px[x, y] < 1 else ' ' for x in range(
            x_beg, x_beg+CHARACTER_WIDTH)]) for y in range(0, CHARACTER_HEIGHT)], ]
    return result


def load_training_letters(fname):
    letter_images = load_letters(fname)
    return {TRAIN_LETTERS[i]: letter_images[i] for i in range(0, len(TRAIN_LETTERS))}

# Reused the function from Part1 to get training words


def read_data(fname='bc.train'):
    exemplars = []
    file = open(fname, 'r')
    for line in file:
        data = tuple([w for w in line.split()])
        exemplars += data

    return exemplars


def return_computed_prob(wordsData):
    transProbDict = {}
    total_pairs = 0
    initial_Prob = {}

    total_words = len(wordsData)

    for ch1 in TRAIN_LETTERS:
        for ch2 in TRAIN_LETTERS:
            transProbDict[(ch1, ch2)] = 0

    for ch in TRAIN_LETTERS:
        initial_Prob[ch] = 0

    for word in words_data:
        for i in range(len(word)-1):
            if i == 0:
                initial_Prob[word[i]] += 1
            pair = (word[i], word[i+1])
            transProbDict[pair] += 1
            total_pairs += 1

    for pair, value in transProbDict.items():
        transProbDict[pair] = value/total_pairs

    for ch, value in initial_Prob.items():
        initial_Prob[ch] = value/total_words

    # print('transProbDict22', transProbDict)
    # print('initial_Prob22', initial_Prob)

    return (transProbDict, initial_Prob)

def viterbi(word, letters_list, initialProb, transitionProb, emissionProb):
        '''
        This method is reused from Part1. It will calculate the maximize the probability of a letter
        given the pixel input from the image.
        '''

        # to hold the values of all the states
        viterbi_dict_map = [{}]


        for letter in letters_list:
            max_first_p = 0
            max_first_PS = ''
            for (key, value) in emissionProb.items():
                temp = 1 - abs(word[0][key] - value)
                if temp > max_first_p:
                    max_first_p = temp
                    max_first_PS = key

            # print('max_first_here11', max_first_p, max_first_PS)


            viterbi_dict_map[0][letter] = { 'computed_prob': initialProb[letter]*max_first_p, 'prevPS': max_first_PS }

        # print('viterbi_dict_map', viterbi_dict_map)

        # iterating over the rest of the chars in the word to get corresponding max probabilities 
        for index in range(1, len(word)):
            viterbi_dict_map.append({})

            for letter in letters_list:
                # initializing the max transition probability and selected Part of speech with the first item
                max_trans_prob = viterbi_dict_map[index-1][letters_list[0]]['computed_prob'] * transitionProb[(letters_list[0],letter)]
                prev_Sel = letters_list[0]

                for l in letters_list[1:]:
                    trans_Prob = viterbi_dict_map[index-1][l]['computed_prob'] * transitionProb[(l, letter)] 
                    if trans_Prob > max_trans_prob:
                        max_trans_prob = trans_Prob
                        prev_Sel = l

                max_p = 0
                for (key, value) in emissionProb.items():
                    temp = 1 - abs(word[index][key] - value)
                    if temp > max_p:
                        max_p = temp
                
                # maximizing the product of transition and emission values and storing it for further
                max_Prob = max_trans_prob * max_p
                viterbi_dict_map[index][letter] = { 'computed_prob': max_Prob, 'prevPS': prev_Sel }

        # print('viterbi_dict_map333', viterbi_dict_map)
        
        outputSeq = []
        max_Prob = 0.0
        best_letter = ' '

        for (l, value) in viterbi_dict_map[-1].items():
            if value['computed_prob'] > max_Prob:
                max_Prob = value['computed_prob']
                best_letter = l

        outputSeq.append(best_letter)

        prev = best_letter
        for y in range(len(viterbi_dict_map)- 2, -1, -1):
            outputSeq.insert(0, viterbi_dict_map[y+1][prev]['prevPS'])
            prev = viterbi_dict_map[y+1][prev]['prevPS']

        return outputSeq


#####
# main program
if len(sys.argv) != 4:
    raise Exception(
        "Usage: python3 ./image2text.py train-image-file.png train-text.txt test-image-file.png")

(train_img_fname, train_txt_fname, test_img_fname) = sys.argv[1:]
train_letters = load_training_letters(train_img_fname)
# print('train_letters111', train_letters)

emissionProbDict = {}

for letter, cList in train_letters.items():
    starCount = 0
    spaceCount = 0
    for ch in cList:
        for c in ch:
            if c == '*':
                starCount += 1
            else:
                spaceCount += 1

    emissionProbDict[letter] = (1 * \
        (starCount / (CHARACTER_WIDTH * CHARACTER_HEIGHT)))
    + (1*(spaceCount / (CHARACTER_WIDTH * CHARACTER_HEIGHT)))


# print('emissionProbDict555', emissionProbDict)


test_letters = load_letters(test_img_fname)
# print('test_letters111', test_letters)
# print("\n".join([ r for r in test_letters[2] ]))
resDict2 = {}
resDict = {}
cnt = -1
#We have given a higher weight to stars , medium weight to spaces and the least weight to noise
weights = [0.7, 0.2, 0.1]

#We loop through each pixel(for * and space) for every letter in the test data and check it with the with the pixel at the corresponding place for all the characters in the training data  and if they are the same
for l in test_letters:
    resDict = {}
    for pPos in range(len(l)):
        maxCnt = -9999
        for letter, pxList in train_letters.items():
            for iPos in range(len(pxList[pPos])):
                #Then we check if the character matches with * or space. if it doesnt match with any of these , the we consider that pixel as noise
                if pxList[pPos][iPos] == l[pPos][iPos]:
                    #We then increase the count for the correspinding match
                    if pxList[pPos][iPos] == "*":
                        if letter not in resDict:
                            resDict[letter] = [1, 0, 0]
                        else:
                            resDict[letter][0] += 1
                    elif pxList[pPos][iPos] == " ":
                        if letter not in resDict:
                            resDict[letter] = [0, 1, 0]
                        else:
                            resDict[letter][1] += 1
                else:
                    if letter not in resDict:
                        resDict[letter] = [0, 0, 1]
                    else:
                        resDict[letter][2] += 1
    cnt = cnt+1
    resDict2[cnt] = resDict

test_dict_prob = {}
for ele in resDict2:
    test_dict_prob[ele] = {}
    for ele2, ls in resDict2[ele].items():
        #The we multiply each cout with their correspoinding weights and then divide that with the total number of pixels to find the probability
        ls[0], ls[1], ls[2] = ls[0]*weights[0], ls[1] * \
            weights[1], ls[2]*weights[2]
        val = (ls[0]+ls[1]+ls[2])/(14*25)
        test_dict_prob[ele][ele2] = val

# print('FRES666', test_dict_prob[4])

fRes = test_dict_prob.copy()
#We the extract the maximum proability and the corresponding letter for that probability
for ele, valL in fRes.items():
    sortedResDict = sorted(valL.items(), key=lambda i: i[1], reverse=True)
    fRes[ele] = sortedResDict[0][0]
# print("outputtt", fRes)

resStr = ""
for ele in fRes:
    resStr = resStr+fRes[ele]

words_data = read_data()

(transProbDict, initial_Prob) = return_computed_prob(words_data)

# print('test_dict_prob222', test_dict_prob[0]['t'])
for i in fRes:
    if fRes[i]=="1":
        fRes[i]=="l"

oLen=len(fRes)
input_viterbi = []
temp = []
for index, item in test_dict_prob.items():
    if fRes[index] == " ":
        input_viterbi.append(temp)
        temp = []
    temp.append(item)

input_viterbi.append(temp)

viterbi_result = []
counter = 0
for i in range(len(input_viterbi)):
    output = viterbi(input_viterbi[i], TRAIN_LETTERS, initial_Prob, transProbDict, emissionProbDict)
    y = len(output)
    while (y > 0):
        viterbi_result.append(fRes[counter])
        counter += 1
        y -= 1



resList2 = ''.join(viterbi_result)


# The final two lines of your output should look something like this:
print("Simple: " + resStr)
print("   HMM: " + resList2)
