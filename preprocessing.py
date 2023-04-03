import pathlib

import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_selection import mutual_info_regression

#Adjust to your computer
basePath = "C:\\Users\\evana\\OneDrive\\Documents\\Courses\\Spring 2023\\CMM\\Final Project\\"
pathToData = basePath + "BR0_data.csv"
pathToMetadata = basePath + "BR0_meta.csv"

#Global Variables
NUMERIC_COLS = ['Age', 'Grade', 'T_Stage', 'N_Stage', 'DRFS_Event', 'DRFS_Year']
Y_COL_NAME = 'ChemoResponse'
RANDOM_STATE = 42

#########################
# General Functionality #
#########################

def loadData(path=pathToData, showHead=False, col0IsIndex=True):
    if col0IsIndex:
        df = pd.read_csv(path, index_col=0)
    else:
        df = pd.read_csv(path)
    if showHead:
        print(df.head())
    return df

def permuteRows(df):
    np.random.shuffle(df[0].to_numpy())
    return df

def combineMetaAndGenomic(regDf, metaDf, onlyYCol=False):
    '''
    Joins the metadata and regular data, only includes the Y value if onlyYCol is True
    '''

    df = pd.merge(regDf, metaDf, left_index=True, right_index=True)
    if onlyYCol:
        regCols = list(regDf.columns)
        regCols.append(Y_COL_NAME)
        df = df[regCols]
    return df

##########################
# Code to Parse Metadata #
##########################

def adjustMetaStrReplacement(metaDf):
    '''
    Gets rid of extra info in the string columns
    '''
    metaDf['PAM50_Class'] = metaDf['PAM50_Class'].apply(lambda x: x.replace('pam50_class: ', ''))
    metaDf['DRFS_Event'] = metaDf['DRFS_Event'].apply(lambda x: x.replace('drfs_1_event_0_censored: ', ''))
    metaDf['DRFS_Year'] = metaDf['DRFS_Year'].apply(lambda x: x.replace('drfs_even_time_years: ', ''))

    metaDf['N_Stage'] = metaDf['N_Stage'].apply(lambda x: x.replace('N', ''))
    metaDf['T_Stage'] = metaDf['T_Stage'].apply(lambda x: x.replace('T', ''))

    return metaDf

def adjustDTypes(metaDf, numericCols):
    '''
    Adjusts the stringed ints and floats back to proper data types, since unknowns have been parsed
    '''

    for col in numericCols:
        metaDf[col] = pd.to_numeric(metaDf[col])

    return metaDf

def parseUnknowns(metaDf):
    '''
    For every column with an unknown as a value,
    create a new column that is a binary indicator of the unknown status
    '''

    for col in metaDf.columns:
        colVals = metaDf[col].unique()
        if "unknown" in colVals:
            uColTitle = str(col) + '_IsUnknown'
            metaDf[uColTitle] = metaDf[col]
            metaDf[uColTitle] = metaDf[uColTitle].apply(lambda x: ifUnknown(x))
            if str(col) == 'Age':
                metaDf[col] = metaDf[col].apply(lambda x: unknownVal(x))
            elif str(col) != 'PAM50_Class':
                metaDf[col] = metaDf[col].apply(lambda x: unknownVal(x, stringIfy=True))

    return metaDf

def ifUnknown(val):
    return 1 if val == 'unknown' else 0

def unknownVal(val, stringIfy=False):
    if stringIfy:
        return '-1' if val == 'unknown' else val
    else:
        return -1 if val == 'unknown' else val

def applyDictMappings(metaDf):
    '''
    Applys the described dictionary mappings of value transformations
    '''

    er_pr_her_Dict = {'positive': 1, 'negative': 0, '-1': -1}
    chemoResp_Dict = {'Resistant': 0, 'Sensitive': 1}

    metaDf['ER_Status'].replace(er_pr_her_Dict, inplace=True)
    metaDf['PR_Status'].replace(er_pr_her_Dict, inplace=True)
    metaDf['HER2_Status'].replace(er_pr_her_Dict, inplace=True)

    metaDf['ChemoResponse'].replace(chemoResp_Dict, inplace=True)

    return metaDf

def figureOutUniques(df):
    for col in df.columns:
        print("Unique values in %s column: "%col)
        print(df[col].unique())

#MAIN METADATA PARSING FUNC
def adjustMetaData(showHead=False, showUniques=False, onlyYCol=False):
    '''
    Main Functionality for parsing metadata
    Replaces NaN's with unknowns, parses extra string info down to the core info,
    Splits off unknown values into indicator columns, adjusts data types,
    maps categorical and numeric info, and one-hot-encodes categorical info
    '''
    metaDf = loadData(pathToMetadata, showHead)
    if onlyYCol:
        chemoResp_Dict = {'Resistant': 0, 'Sensitive': 1}
        metaDf['ChemoResponse'].replace(chemoResp_Dict, inplace=True)
        if showUniques:
            figureOutUniques(metaDf)
        return metaDf

    metaDf.fillna('unknown', inplace=True)
    metaDf = adjustMetaStrReplacement(metaDf)
    metaDf = parseUnknowns(metaDf)
    metaDf = adjustDTypes(metaDf, NUMERIC_COLS)
    metaDf = applyDictMappings(metaDf)

    metaDf = pd.get_dummies(metaDf, prefix=['PAM50_Class'], columns=['PAM50_Class'])
    if showUniques:
        figureOutUniques(metaDf)
    return metaDf

##############################
# Code to Parse Regular Data #
##############################

#MAIN RegularDATA PARSING FUNC
def adjustRegularData(showHead=False, useNegatives=True):
    regDf = loadData(pathToData, showHead)
    regDf = scaleValues(regDf, useNegatives)

    return regDf

def scaleValues(regDf, useNegatives=True):
    '''
    Scales all columns with a negative value to [-1,1], and all with only positive values to [0,1]
    '''
    for col in regDf.columns:
        colMin = regDf[col].min()
        colMax = regDf[col].max()
        if colMin < 0 and useNegatives:
            regDf[col] = regDf[col].apply(lambda x: neg1to1scale(x, colMin, colMax))
        else:
            regDf[col] = regDf[col].apply(lambda x: zero1scale(x, colMin, colMax))

    return regDf

def neg1to1scale(myVal, valsMin, valsMax):
    return 2*zero1scale(myVal, valsMin, valsMax) - 1

def zero1scale(myVal, valsMin, valsMax):
    return (myVal - valsMin)/( valsMax - valsMin)

#Turns out our DF has negative values. Only in some cols tho.
def checkMin(df):
    df2 = df.stack()
    print("DF Minimum: %f" %df2.min())
    print()

######################
# Mutual Information #
######################

def nonYNpArray(df):
    '''
    Get the array of all non-y values in the given matrix.
    '''
    cols = list(df.columns)
    cols = [c for c in cols if c != Y_COL_NAME]
    dfAsArray = df[cols].to_numpy()
    return dfAsArray

def mutualInformation(df, yColName):
    '''
    Goes through each gene, and calculates its mutual information with our output Y
    prints out the indices of the gene's w. the largest mutual informations,
    and their mutual info vals.

    Used for continuous variables, not discrete.
    '''
    yvals = np.array( df[yColName] )
    dfAsArray = nonYNpArray(df)
    mutualInfo = mutual_info_regression(dfAsArray, yvals)

    return mutualInfo

def reconstructDataFrame(npArray, indicesForRows, YVals, colNames=None):
    '''
    Given a numpy array, reconstruct the pd dataframe associated with it.
    Column names are only useful if you're using mutual information.
    '''

    if colNames is None:
        colNames = list(range(npArray.shape[0] ))
        colNames = ['Feature_%d'%c for c in colNames]

    outDf = pd.DataFrame(npArray, columns=colNames)
    outDf['Index Col'] = indicesForRows
    outDf[Y_COL_NAME] = YVals
    outDf = outDf.set_index('Index Col')
    return outDf

def getSubsample(numXs, df, allMutualInfoVals=False):
    '''
    Either uses mutual information or SVD to do dimensionality reduction.
    If using mutual information, assumes you input infoVals, the output of mutualInformation

    #######################
    Run on the GENOMIC Data
    #######################
    '''

    indicesForRows = list(df.index.values)
    YVals = list(df[Y_COL_NAME])

    cols = list(df.columns)
    cols = [c for c in cols if c != Y_COL_NAME]

    if allMutualInfoVals is not False:
        nonYArray = nonYNpArray(df)
        colsIndicesToKeep = np.argpartition(allMutualInfoVals, -numXs)[-numXs:]
        nonYArrayT = nonYArray.T
        outArr = nonYArrayT[colsIndicesToKeep]
        outArr = outArr.T
        colNames = np.array(cols)[colsIndicesToKeep]
        miniDf = reconstructDataFrame(outArr, indicesForRows, YVals, colNames)

    else:
        nonYArray = nonYNpArray(df)
        svd = TruncatedSVD(n_components=numXs, random_state=RANDOM_STATE)
        outArr = svd.fit_transform( nonYArray )
        svdNames = [cols[i] for i in svd.components_[0].argsort()[::-1]]
        svdNames = np.array(svdNames)[:numXs]
        miniDf = reconstructDataFrame(outArr, indicesForRows, YVals, colNames=svdNames)

    return miniDf

##################
# Final Function #
##################

def generateDf(includeMeta=True, useMutualInfo=True, numXs=10):
    '''
    includeMeta - include metadata, True or False
    useMutualInfo - use mutual information or truncatedSVD, True or False
    numXs - the dimension you're reducing the GENOMIC ONLY Data To, Int

    Returns a pandas df with the y col
    '''
    regDf = adjustRegularData(showHead=False)

    if includeMeta:
        metaDf = adjustMetaData(showHead=False, onlyYCol=False)
        miniMetaDf = metaDf[Y_COL_NAME]
        regDfPlusY = combineMetaAndGenomic(regDf, miniMetaDf, onlyYCol=False)
    else:
        metaDf = adjustMetaData(showHead=False, onlyYCol=True)
        miniMetaDf = metaDf[Y_COL_NAME]
        regDfPlusY = combineMetaAndGenomic(regDf, miniMetaDf, onlyYCol=True)

    if useMutualInfo:
        mf = mutualInformation(regDfPlusY, Y_COL_NAME)
        miniDf = getSubsample(numXs=numXs, df=regDfPlusY, allMutualInfoVals=mf)

    else:
        miniDf = getSubsample(numXs=10, df=regDfPlusY, allMutualInfoVals=False)

    if not includeMeta:
        return miniDf

    else:
        fullDf = combineMetaAndGenomic(miniDf, metaDf, onlyYCol=False)
        return fullDf


if __name__ == "__main__":
    '''
    metaDf = adjustMetaData(showHead=False, onlyYCol=False)
    miniMetaDf = metaDf[Y_COL_NAME]
    regDf = adjustRegularData(showHead=False)
    regDfPlusY = combineMetaAndGenomic(regDf, miniMetaDf, onlyYCol=True)
    mf = mutualInformation(regDfPlusY, Y_COL_NAME)
    miniDf = getSubsample(numXs=10, df=regDfPlusY, allMutualInfoVals=mf)
    #miniDf = getSubsample(numXs=10, df=regDfPlusY, allMutualInfoVals=False)
    fullDf = combineMetaAndGenomic(miniDf, metaDf, onlyYCol=False)

    print(fullDf.head())
    '''

    """
    Note: Its significantly faster do NOT Do the preprocessing repeatedly, eg not use 
    generateDf but instead use the commented out code above and run multiple diff numXs at once. 
    """

    df1 = generateDf(includeMeta=True, useMutualInfo=True, numXs=10)
    df2 = generateDf(includeMeta=True, useMutualInfo=False, numXs=10)
    df3 = generateDf(includeMeta=False, useMutualInfo=True, numXs=10)
    df4 = generateDf(includeMeta=False, useMutualInfo=False, numXs=10)