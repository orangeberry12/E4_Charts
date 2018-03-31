import csv
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import math


def openReadFile(filepath):
	"""Opens .csv file and returns data in 2D array: [row][column]"""
	dataArray = []
	y = [] #E4
	x = [] #time
	with open(filepath, newline='') as f:
		reader = csv.reader(f)
		for row in reader:
			dataArray.append(row)

	return dataArray


def getAverage(yVals):
	"""
	Computes the average of all values in an array.
	"""
	sum = 0
	for y in yVals:
		sum += y
	avg = sum/len(yVals)
	return avg

def polyAvg(x,y,deg):
	"""Computes the best fit 3rd degree polynomial"""
	c = np.polyfit(x,y,deg) #coefficients
	poly3 = np.poly1d(c) #create degree three polynomial function

	return poly3(x)


def remap(inMin, inMax, outMin, outMax, array):
	"""
	Takes an array, it's minimum and maximum, and remaps 
	to a new range [outMin, outmax]
	"""
	remVals=[]
	for i in range(len(array)):
		remapped = ((array[i]-inMin)/(inMax-inMin))*(outMax-outMin)+outMin 
		remVals.append(remapped)
	return remVals


def getMinMax(data):
	"""
	Takes list in format from SeparateData
	[relaxX,relaxY, trans1X, trans1Y, habitX,habitY, trans2X, trans2Y, testX,testY, minY, maxY]
	Updates min and max.
	"""
	allY = []
	for i in range(1,10,2):
		allY += data[i]

	minY = min(allY)
	maxY = max(allY)

	data[10] = minY
	data[11] = maxY

	return data


def clean(x,y):
	"""
	Assumes lists x and y of same length
	"""
	newX = []
	newY = []
	print("length of y is: " + str(len(y)))
	print("length of x is: " +str(len(x)))
	for i in range(len(y)):
		if (y[i]>1.5) and (y[i]<40):
			newX.append(x[i])
			newY.append(y[i])
	return [newX, newY]


def clean2(x,y):
	avg = getAverage(y)
	sigma = standardDeviation(y)
	allowRange = 2*sigma #allowable deviation from average
	newX =[]
	newY = []
	for i in range(len(y)):
		if (y[i]< avg+allowRange) and (y[i] > avg-allowRange):
			newX.append(x[i])
			newY.append(y[i])
	return[newX,newY,avg, allowRange]


def standardDeviation(values):
	"""
	Returns standard of deviation for a list of values.
	"""
	theMean = getAverage(values)
	squareDifference = []
	for v in values:
		squareDifference.append((v-theMean)**2)

	avgSD = getAverage(squareDifference)
	sigma = math.sqrt(avgSD)

	return sigma


def separateData(data,tagdata):
	"""
	Separates data into relax period, habituation period, test period
	There should be 6 time tags marking the start/end of each period.
	Correct the tags file if it is wrong.
	"""
	timeStart = float(data[0][0])	#in UTC
	sampleRate = float(data[1][0]) #in Hz
	temp = data[3:] # the first value is always junk


	#clean up tags array
	tags=[]
	for i in range(len(tagdata)):
		tags.append(float(tagdata[i][0]))

	#initialize x and y arrays
	y=[]
	times=[]

	#get all the times and y-vals
	for d in range(len(temp)):
		y.append(float(temp[d][0]))
		times.append(timeStart+(d/sampleRate))

	#get indices for three periods
	divIndex = []
	x =[] 			#initialize x-axis markers
	current = 0		#counter to keep track of position in time

	for t in range(6):
		print("t is: " + str(t))
		for s in range(current, len(times)):
			#x.append(s)		#populating x values
			#find all the index to separate data
			if (times[s]>tags[t]):
				print("found one!")
				divIndex.append(s)
				current = s
				break
			elif (s == len(times)-1) and (t==5):
				divIndex.append(s)

	print(divIndex)
	#make x-coordinates
	for r in range(len(y)):
		x.append(r)


	#divide data into separate lists
	relaxX = x[divIndex[0]:divIndex[1]]
	relaxY = y[divIndex[0]:divIndex[1]]
	habitX = x[divIndex[2]:divIndex[3]]
	habitY = y[divIndex[2]:divIndex[3]]
	testX = x[divIndex[4]:divIndex[5]]
	testY = y[divIndex[4]:divIndex[5]]

	print("relaxX is length: " +str(len(relaxX)))

	#transition periods
	trans1X = x[divIndex[1]:divIndex[2]]
	trans1Y = y[divIndex[1]:divIndex[2]]
	trans2X = x[divIndex[3]:divIndex[4]]
	trans2Y = y[divIndex[3]:divIndex[4]]

	minY = min(y)
	maxY = max(y)
	print("minY is: " + str(minY))
	print("maxY is: " +str(maxY))

	return [relaxX,relaxY, trans1X, trans1Y, habitX,habitY, trans2X, trans2Y, testX,testY, minY, maxY]

def cleanSessions(data):
	cleanData = []
	minY =0
	maxY =0
	for i in range(0,len(data)-3,2):
		print("i is: " +str(i))
		print(len(data[i]))
		print(len(data[i+1]))
		temp = clean2(data[i],data[i+1])
		cleanData.append(temp[0])
		cleanData.append(temp[1])
		#set new min and max Y
		if (i==0): # the first session (relax)
			minY = min(data[i+1])
			maxY = max(data[i+1])
		else:
			if (min(data[i+1])<minY):
				minY = min(data[i+1])
			if (max(data[i+1])>maxY):
				maxY = max(data[i+1])
		print("maxY is: " + str(maxY))
		print("minY is: " +str(minY))
	
	cleanData.append(minY)
	cleanData.append(maxY)

	return cleanData



def rescaleData (data,datatype):
	"""
	Takes array from separateData and rescales based on sampling rate (EDA/HR)
	Rescales transitions to 3mins
	Rescales Relax to 15 min
	Rescales Habituation to 10 min
	Rescales  Stress to 15 min
	Rescales y-values between 0-1
	"""

	if (datatype == "EDA"):	#4hz
		timeRange = [[0,3600],[3600,4320],[4320,6720],[6720,7440],[7440,11040]]

	elif (datatype == "HR") or (datatype == "ACC"): #1hz
		timeRange = [[0,900],[900,1080],[1080,1680],[1680,1860],[1860,2760]]

	minY = data[len(data)-2]
	maxY = data[len(data)-1]
	print("minY is in rescale is: " + str(minY))
	print("maxY is in rescale is: " + str(maxY))

	for i in range(0,len(data)-2,2):		# remap x values
		print("i is: " + str(i))
		print("length data[i] is: " +str(len(data[i])))
		dL = len(data[i])-1
		print("dL is: " + str(dL))
		data[i] = remap(data[i][0], data[i][dL],timeRange[int(i/2)][0],timeRange[int(i/2)][1],data[i])

	for q in range(1,len(data)-2,2):
		print("q is: " + str(q))
		print("length data[q] is: " +str(len(data[q])))		# remap y values
		dL = len(data[q])-1
		data[q] = remap(minY, maxY,0,1,data[q])
		#print(data[i])

	return data


def setXYRange(groupFile, type):
	"""
	Returns a master list [[x1], [x2], [x3],...],[[y1],[y2],[y3],...]
	from relax, habit, and stress sets of data for a group (fast, slow or control)
	groupFile = csv containing subject id numbers
	type = the type of data you are interested in graphing (EDA, ACC, HR)
	"""
	masterX=[]
	masterY=[]
	subjects = openReadFile(groupFile)

	for s in range(len(subjects)):
		subjects[s]=subjects[s][0]

	for s in subjects:
		tagPath = "./"+s+"/"+"E4"+"/"+"tags.csv"

		#Now get the right path to the datafile.
		if (type == "EDA"):
			dataPath = "./"+s+"/"+"E4"+"/"+"EDA.csv"
		elif (type == "hr" or type == "HR"):
			dataPath = "./"+s+"/"+"E4"+"/"+"HR.csv"
		elif (type == "acc" or type == "ACC"):
			dataPath = "./"+s+"/"+"E4"+"/"+"ACC.csv"

		tagData = openReadFile(tagPath)
		sensorData = openReadFile(dataPath)
		data = separateData(sensorData,tagData)
		cleanData = cleanSessions(data)
		updateData = getMinMax(cleanData)

		#scale x and y axis
		scaleData = rescaleData(updateData,type)
		#scaleData = data
		#print(scaleData)

		concatX = []
		concatY = []
		#all relevant x,y concat into one
		for i in range(0,len(scaleData)-2,2):
			concatX += scaleData[i]
			concatY += scaleData[i+1]
			print(scaleData[i+1])
		
		#print(concatY)
		masterX.append(concatX)
		masterY.append(concatY)


	return masterX, masterY



def plotMultiGraph(groupFile, type, group):
	"""
	groupFile = array of filepaths containing members of group
	type = "EDA" or "HR"
	group = array of group names ["Control", "Slow", "Fast"]
	"""
	for g in range(len(groupFile)):	#for each group
		fig, ax = plt.subplots()
		linecolors = ["#6AC8C7","#FF8F4B","#F1C62F","#3BAF51","#456AAD","#9C69C4","#F061C9","#E95252"] 

		data = setXYRange(groupFile[g],type)
		masterX = data[0]
		masterY = data[1]

		for i in range(len(masterX)):
			#get x,y for first subject in group
			x=masterX[i]
			y=masterY[i]

			#now generate polynomial average
			p3 = polyAvg(x,y,4)

			#plot the trend line.
			plt.plot(x,p3, color = linecolors[i], linestyle = "-", linewidth = 4, zorder = 3) 
			plt.plot(x,y, color = linecolors[i], linestyle = "-", linewidth = 0.5, alpha = 0.5, zorder = 2)

		if ((type=="eda") or (type == "EDA")):
			#Range of values for axes of graph
			xRange = 11040
			yLabel = "ΔEDA (μS)"
			#x-coordinate for vertical lines
			v1 = 3600
			v2 = 4320
			v3 = 6720
			v4 = 7440

		else:
			#x-coordinate for vertical lines
			v1 = 900
			v2 = 1080
			v3 = 1680
			v4 = 1860

			if ((type =="hr") or (type == "HR")):
				#Range of values for axes for graph
				xRange = 2760
				yLabel = "ΔBR (bpm)"

		ax.spines['top'].set_visible(False)
		ax.spines['right'].set_visible(False)
		#ax.spines['bottom'].set_visible(False)
		#ax.spines['left'].set_visible(False)

		#plot axes, title,etc
		plt.xlim(0,xRange)
		plt.ylim(-.1,1.1)
		plt.ylabel(yLabel)
		plt.xlabel("Time")
		plt.axvline(x = v1, color = "255", linestyle = "--", linewidth = 0.5, zorder=4)
		plt.axvline (x = v2, color = "255", linestyle = "--", linewidth = 0.5, zorder=4)
		plt.axvline(x = v3, color = "255", linestyle = "--", linewidth = 0.5, zorder=4)
		plt.axvline (x = v4, color = "255", linestyle = "--", linewidth = 0.5, zorder=4)
		
		
		
		#plot horizontal lines at y values
		for y in range(-1,7,1):
			plt.axhline(y*0.25, color = "80", linewidth = "0.25", linestyle="-", zorder = 1)
		
		plt.title(type + " Trend: " + group[g] + " Group")

		#create and save plot figure.
		fig = matplotlib.pyplot.gcf()
		fig.set_size_inches(10,4)
		fig.savefig("./E4_"+group[g]+"-"+type+'_Final.png', dpi = 300)
		plt.clf() #clear figure for next group.

	return


def plotBarChart(groupFile, type, group):
	"""
	groupFile = array of filepaths to participant lists
	type = type of data to be visualized ("EDA", "HR", "ACC")
	group = array of group names ["Control", "Slow", "Fast"]
	"""

	for g in range(len(groupFile)):

		"""
		Set some plot style stuff -----------------------------------------------------
		"""
		#get rid of border around graphs
		fig, ax = plt.subplots()
		ax.spines['top'].set_visible(False)
		ax.spines['right'].set_visible(False)
		#ax.spines['bottom'].set_visible(False)
		#ax.spines['left'].set_visible(False)

		#colors control, fast, slow
		if (group[g] == "Control"):
			barcolor = "#6AC8C7"
		elif (group[g] == "Fast"):
			barcolor = "#F1C62F"
		elif (group[g] == "Slow"):
			barcolor= "#FF8F4B"
		elif(group[g] == "HR" or group == "BR"):
			barcolor = ["#6AC8C7","#FF8F4B","#F1C62F"]

		"""
		Get the data--------------------------------------------------------------------
		"""
		subjects = openReadFile(groupFile[g])

		for s in range(len(subjects)):
			subjects[s]=subjects[s][0]

		subjectDict = {}
		avgDifference =[]
		yUnits =""

		for s in subjects:
			tagPath = "./"+s+"/"+"E4"+"/"+"tags.csv"

			#Now get the right path to the datafile.
			if ((type == "eda") or (type == "EDA")):
				dataPath = "./"+s+"/"+"E4"+"/"+"EDA.csv"
				yUnits = "ΔEDA (μS)"
			elif (type == "hr" or type == "HR"):
				dataPath = "./"+s+"/"+"E4"+"/"+"HR.csv"
				yUnits = "ΔHR (bpm)"

			tagData = openReadFile(tagPath)
			sensorData = openReadFile(dataPath)
			data = separateData(sensorData,tagData)
			cleanData = cleanSessions(data)
			updateData=getMinMax(cleanData)
			scaleData = rescaleData(updateData,type)

			rAvg = getAverage(scaleData[1]) #check separateData for order
			sAvg = getAverage(scaleData[9]) #check separateData for order
			diff =sAvg-rAvg
			avgDifference.append(diff)
			#dictionary subject: change in eda
			subjectDict[s] = diff

		#Sort the subject dictionary by change in EDA
		sortedSD = [(k,subjectDict[k]) for k in sorted(subjectDict, key = subjectDict.get, reverse=False)]
		sortedSD = dict(sortedSD)
		
		"""
		Y-range and horizontal lines-------------------------------------------------
		#Values have been normalized between 0 and 1. Yay!
		"""
		yMax = 1
		yMin =0
		for y in range(yMin,11,1):
			plt.axhline(y*0.1, color = "80", linewidth = "0.25", linestyle="-", zorder = 1)
		
		"""
		Plot bars--------------------------------------------------------------------
		"""
		x = []
		for i in range(len(avgDifference)): x.append(i)
		#bar plot
		#plt.bar(x,avgDifference, width = 0.5, align='center', color = barcolor, zorder = 2)
		plt.bar(range(len(sortedSD)),sortedSD.values(), width = 0.5, align='center', color = barcolor, zorder = 2)

		"""
		Labels and stuff--------------------------------------------------------------
		"""
		plt.ylim(yMin,yMax)
		plt.tick_params(top='off', bottom='off', left='off', right='off', \
			labelleft='on', labelbottom='on')
		plt.xticks(range(len(sortedSD)), sortedSD.keys())

		plt.ylabel(yUnits)
		plt.xlabel("PARTICIPANTS")

		if ("HR" in type):
			plt.title("Average Difference in Heart Rate from Relaxed to Stressed Conditions\n\n" + group[g] + " Group")
		elif ("EDA" in type):
			plt.title("Average Difference in EDA from Relaxed to Stressed Conditions\n\n" + group[g] + " Group")


		fig = matplotlib.pyplot.gcf()
		fig.savefig("./"+"E4_BarChart_"+type+"_"+group[g]+"_v3.png", dpi = 300)
		plt.clf() # clear plot for next group.

	return


def singleBarChart(groupFile, type, group):
	"""
	Set some plot style stuff -----------------------------------------------------
	"""
	#get rid of border around graphs
	fig, ax = plt.subplots()
	ax.spines['top'].set_visible(False)
	ax.spines['right'].set_visible(False)
	#ax.spines['bottom'].set_visible(False)
	#ax.spines['left'].set_visible(False)

	#colors control, fast, slow
	barcolor = ["#6AC8C7","#FF8F4B","#F1C62F"]

	"""
	Get the data--------------------------------------------------------------------
	"""
	groupAverages = []
	stanDev = []
	for g in groupFile:
		subjects = openReadFile(g)

		for s in range(len(subjects)):
			subjects[s]=subjects[s][0]

		avgDifference =[]
		#subjects per group
		for s in subjects:
			tagPath = "./"+s+"/"+"E4"+"/"+"tags.csv"

			#Now get the right path to the datafile.
			if ((type == "eda") or (type == "EDA")):
				dataPath = "./"+s+"/"+"E4"+"/"+"EDA.csv"
				yLabel = "ΔEDA (μS)"
			elif (type == "hr" or type == "HR"):
				dataPath = "./"+s+"/"+"E4"+"/"+"HR.csv"
				yLabel = "ΔHR (bpm)"
			elif (type == "acc" or type == "ACC"):
				dataPath = "./"+s+"/"+"E4"+"/"+"ACC.csv"

			tagData = openReadFile(tagPath)
			sensorData = openReadFile(dataPath)
			data = separateData(sensorData,tagData)
			cleanData = cleanSessions(data)
			updateData=getMinMax(cleanData)
			scaleData = rescaleData(updateData,type)

			rAvg = getAverage(scaleData[1]) #check separateData for order
			sAvg = getAverage(scaleData[9]) #check separateData for order
			diff =sAvg-rAvg			
			avgDifference.append(sAvg-rAvg)
		
		groupAvg = getAverage(avgDifference)
		groupAverages.append(groupAvg)

		sigmaGroup = standardDeviation(avgDifference)
		stanDev.append(sigmaGroup)
	"""
	Y-range and horizontal lines-------------------------------------------------
	"""
	yMin = 0
	yMax =1
	for y in range(yMin,11,1):
		plt.axhline(y*0.1, color = "80", linewidth = "0.25", linestyle="-", zorder = 1)
	
	"""
	Plot bars--------------------------------------------------------------------
	"""
	x = []
	for i in range(len(groupAverages)): x.append(i)
	#bar plot
	plt.bar(x,groupAverages, width = 0.5, align='center', color = barcolor, zorder = 2)

	"""
	Error bars--------------------------------------------------------------------
	"""
	plt.errorbar(x,groupAverages,stanDev, linestyle="None", color = "150", elinewidth = 0.75, capsize = 3, zorder = 3)

	
	"""
	Labels and stuff--------------------------------------------------------------
	"""
	plt.xticks(range(3), group)

	plt.ylim(yMin,yMax)
	plt.ylabel(yLabel)
	plt.tick_params(top='off', bottom='off', left='off', right='off', \
		labelleft='on', labelbottom='on')

	if ("HR" in type):
		plt.title("Average Difference in Heart Rate from Relaxed to Stressed Conditions\n")
	elif ("EDA" in type):
		plt.title("Average Difference in EDA from Relaxed to Stressed Conditions\n")


	fig = matplotlib.pyplot.gcf()
	#fig.set_size_inches(10,5)
	fig.savefig("./"+"E4_BarChart_"+type+"_averages_v3.png", dpi = 300)
	return



"""
--------------------------------------------------------------------------------------
Run stuff down here
"""
allGroups = ["./EDA-Sample.csv"]
groupNames = ["All"]
#allGroups = ["./PARTICIPANT LIST - Control.csv", "./PARTICIPANT LIST - Slow.csv", "./PARTICIPANT LIST - Fast.csv"] 
#groupNames = ["Control", "Slow", "Fast"]

plotMultiGraph(allGroups,"EDA", groupNames)
#plotBarChart(allGroups,"EDA", groupNames)
#singleBarChart(allGroups,"EDA", groupNames)

