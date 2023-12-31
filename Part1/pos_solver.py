###################################
# CS B551 Fall 2022, Assignment #3
#
# Your names and user ids: hgalla-rathlur-vchitti
#
# (Based on skeleton code by D. Crandall)


import random
import math
from collections import Counter


# We've set up a suggested code structure, but feel free to change it. Just
# make sure your code still works with the label.py and pos_scorer.py code
# that we've supplied.
#
class Solver:
    def __init__(self):
        self.classNames=['det','noun','adj','verb','adp','adv','conj','prt','pron','num','x','.']
        self.sProbDict={}
        self.initialProb = {}
        self.transProbDict = {}


    
    # Calculate the log of the posterior probability of a given sentence
    #  with a given part-of-speech labeling. Right now just returns -999 -- fix this!
    def posterior(self, model, sentence, label):
        if model == "Simple":
            simLogProb = 0
            for index in range(len(sentence)):
                eValue = 1e-12
                if sentence[index] in self.sProbDict:
                    eValue = self.sProbDict[sentence[index]][label[index]] or 1e-12

                simLogProb += math.log(eValue)

            return simLogProb

        elif model == "HMM":
            hmmLogProbs = 0
            for index in range(len(sentence)):
                eValue = 1e-10
                if sentence[index] in self.sProbDict:
                    eValue = self.sProbDict[sentence[index]][label[index]] or 1e-10

                hmmLogProbs += math.log(eValue)

            for i in range(1, len(label)):
                lastPos = label[i-1]
                eValue = 1e-10
                if sentence[i] in self.sProbDict:
                    eValue = self.transProbDict[(lastPos, label[i])] or 1e-10
                
                hmmLogProbs += math.log(eValue)

            return hmmLogProbs

        elif model == "Complex":
            complexLogProbs = 0

            for i in range(1, len(sentence)):
                eValue = 1e-11

                if sentence[i-1] in self.sProbDict:
                    eValue = self.sProbDict[sentence[i-1]][label[i-1]] or 1e-11

                complexLogProbs += math.log(eValue)

                eValue = 1e-11

                if sentence[i] in self.sProbDict:
                    eValue = self.sProbDict[sentence[i]][label[i-1]] or 1e-11

                complexLogProbs += math.log(eValue)

            for j in range(1, len(label)):
                dep1 = label[j-1]
                dep2 = -1
                if j-2 >= 0:
                    dep2 = label[j-2]

                eValue = 1e-11
                if sentence[j] in self.sProbDict:
                    eValue = self.transProbDict[(dep1, label[j])] or 1e-10
                
                complexLogProbs += math.log(eValue)

                if dep2 == -1:
                    continue

                eValue = 1e-11
                if sentence[j] in self.sProbDict:
                    eValue = self.transProbDict[(dep2, label[j])] or 1e-10
                
                complexLogProbs += math.log(eValue)

            return complexLogProbs

    
        else:
            print("Unknown algo!")



    # Do the training!
    def train(self, data):   
        self.computeParameters(data)


    
    def computeParameters(self,data):
        classDict={}
        uniqueWords=set()
        
        # initializing class, initial and transition dictionaries
        for ele in self.classNames:
            classDict[ele]=[]
            self.initialProb[ele] = 0

        for c1 in self.classNames:
            for c2 in self.classNames:
                self.transProbDict[(c1,c2)] = 0

        total_pairs = 0

        # Iterating over the training data to get unique words, POS transitions and other counts
        for ele in data:
            (s,p)=ele
            self.initialProb[p[0]] += 1
            uniqueWords.update(s)

            #ClassDict holds types of POS as the keys and the values contain all the words in reviews belonging to each class
            for pos in range(len(p)):
                classDict[p[pos]].append(s[pos])
            
            # To get transition pairs of parts of speech
            for i in range(len(p)-1):
                pair = (p[i],p[i+1])
                self.transProbDict[pair] += 1
                total_pairs += 1

        # normalizing the transition probability with the total pairs
        for pair,val in self.transProbDict.items():
            self.transProbDict[pair]=val/total_pairs

        # Normalizing the initial probability count
        sentence_count = len(data)
        for key, value in self.initialProb.items():
            self.initialProb[key] = value/sentence_count

        # Initializing the emission probability dictionary 
        probDict={}

        for ele in uniqueWords:
            probDict[ele]={}
        
        # Maintaining the counts for each POS
        cntClassDict={}

        for cl in self.classNames:
            cntClassDict[cl]=Counter(classDict[cl])

        # Calculating the emission probabilities 
        for word in uniqueWords:
            for classes in self.classNames:
                probDict[word][classes]=((cntClassDict[classes][word]))/((len(classDict[classes])))

        self.sProbDict = probDict
        return

    def viterbi(self, sentence, psNames, initialProb, transitionProb, emissionProb):
        '''
        This method calculates the maximum probability for each observed word in the sequence w.r.t the values
        calculated form the previous states. It gives us the most likely sequence of parts of speech.
        Referred to the blog: https://www.pythonpool.com/viterbi-algorithm-python/ to understand and implement the
        algorithm in python.
        '''

        # to hold the values of all the states
        viterbi_dict_map = [{}]


        max_first_p = 0
        max_first_PS = ''

        # iterating over states to find the max initial probability for the first word in the sentence
        for part_s in psNames:
            # setting a low initial value instead of 0
            emmissionValue = 1e-6
            
            if sentence[0] in emissionProb:
                emmissionValue = emissionProb[sentence[0]][part_s]

            if emmissionValue > max_first_p:
                max_first_p = emmissionValue
                max_first_PS = part_s

            viterbi_dict_map[0][part_s] = { 'computed_prob': initialProb[part_s]*emmissionValue, 'prevPS': max_first_PS }

        # iterating over the rest of the words in the sentence to get corresponding max probabilities 
        for index in range(1, len(sentence)):
            viterbi_dict_map.append({})

            for p_speech in psNames:
                # initializing the max transition probability and selected Part of speech with the first item
                max_trans_prob = viterbi_dict_map[index-1][psNames[0]]['computed_prob'] * transitionProb[(psNames[0],p_speech)]
                prev_Sel = psNames[0]

                # iterating over rest of the states of PS and finding the max
                for pS in psNames[1:]:
                    trans_Prob = viterbi_dict_map[index-1][pS]['computed_prob'] * transitionProb[(pS, p_speech)] 
                    if trans_Prob > max_trans_prob:
                        max_trans_prob = trans_Prob
                        prev_Sel = pS

                # finding the emission probability if exists or we give a very small value
                emmissionValue = 1e-5
                if sentence[index] in emissionProb:
                    emmissionValue = emissionProb[sentence[index]][p_speech]
                
                # maximizing the product of transition and emission values and storing it for further
                max_Prob = max_trans_prob * emmissionValue
                viterbi_dict_map[index][p_speech] = { 'computed_prob': max_Prob, 'prevPS': prev_Sel }
        
        # finding the best POS tagging for the last word. 
        # By default we assigned '.' as the last tag, if no such POS exists for the word
        outputSeq = []
        max_Prob = 0.0
        best_PS = '.'

        for (pS, value) in viterbi_dict_map[-1].items():
            if value['computed_prob'] > max_Prob:
                max_Prob = value['computed_prob']
                best_PS = pS

        outputSeq.append(best_PS)

        # Initializing the prev best POS and tracing back thorough the dictionary to find tags for other words.
        prev = best_PS
        for y in range(len(viterbi_dict_map)- 2, -1, -1):
            if prev in viterbi_dict_map[y+1]:
                outputSeq.insert(0, viterbi_dict_map[y+1][prev]['prevPS'])
                prev = viterbi_dict_map[y+1][prev]['prevPS']

            else:
                # if the word doesn't exist in the training data then we assume it as a noun
                outputSeq.insert(0,'noun')
                prev = 'noun'

        return outputSeq

    def sample_values(self, sentence, pSpeechNames, emissionProbDict):
        cal_bias = {}

        for wrd in sentence:
            emissionValues = []

            for p_speech in pSpeechNames:
                if wrd in emissionProbDict:
                    emissionValues.append(emissionProbDict[wrd][p_speech])
                else:
                    emissionValues.append(1e-10)

            cumulated_sum = 0
            for i in range(len(emissionValues)):
                modified_value = (emissionValues[i] / sum(emissionValues))
                cumulated_sum = cumulated_sum + modified_value
                emissionValues[i]= cumulated_sum

            cal_bias[wrd] = emissionValues

        sampled_output = []

        for i in range(len(sentence)):
            sampled_output.append([])
            y = 0
            while (y < 1250):
                random_bias = random.random()
                for j in range(len(pSpeechNames)):
                    if random_bias <= cal_bias[sentence[i]][j]:
                        sampled_output[i].append(pSpeechNames[j])
                        break
                y += 1

        return sampled_output
        

    # Functions for each algorithm. Right now this just returns nouns -- fix this!
    def simplified(self, sentence):
        resList=[]

        for wrd in sentence:
            resOut=[-999999,"n"]
            if wrd in self.sProbDict:
                for ele,value in self.sProbDict[wrd].items():
                    if value>resOut[0]:
                        resOut=[value,ele]

                resList.append(resOut[1])
            else:
                resList.append("noun")
        
        return resList

    def hmm_viterbi(self, sentence):
        viterbi_output=self.viterbi(sentence, self.classNames, self.initialProb, self.transProbDict, self.sProbDict)
        return viterbi_output

    def complex_mcmc(self, sentence):
        sampled_output = self.sample_values(sentence, self.classNames, self.sProbDict)
        cResList = []

        for i in range(len(sampled_output)-1):
            cResList.append(sampled_output[i][len(sampled_output[i]) - 1])

        cResList.append('.')

        return cResList



    # This solve() method is called by label.py, so you should keep the interface the
    #  same, but you can change the code itself. 
    # It should return a list of part-of-speech labelings of the sentence, one
    #  part of speech per word.
    #
    def solve(self, model, sentence):
    
        if model == "Simple":
            return self.simplified(sentence)
        elif model == "HMM":
            return self.hmm_viterbi(sentence)
        elif model == "Complex":
            return self.complex_mcmc(sentence)
        else:
            print("Unknown algo!")

