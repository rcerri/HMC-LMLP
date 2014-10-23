/*
 * Functions used in the HMC-LMLP program
 * Fuctions used by single-label version
 * Not object oriented, although written in Java.
 * That's bad, I know! What a pity! But it works anyway...
 * 
 * Absolutely no guarantees or warranties are made concerning the suitability,
 * correctness, or any other aspect of this program. Any use is at your own risk.
 */
package hmc_lmlp_1hl;

import java.io.*;
import java.util.*;
import java.util.regex.*;

/**
 *
 * @author Ricardo Cerri
 */
public class FunctionsSingleLabel implements Serializable {

    /*===========================================================================
     * Get the outputs of the network applying threshold values
     *===========================================================================*/
    static int getOutputThreshold(double outputActFunc, double threshold,
            double thresholdReductionFactor, int actLevel) {

        int output;

        //Threshold reduction factor
        threshold = threshold * Math.pow(thresholdReductionFactor, actLevel);

        if (outputActFunc > threshold) {
            output = 1;
        } else {
            output = 0;
        }


        return output;
    }

    /*===========================================================================
     * Apply thresholds in matrix of outputs
     *===========================================================================*/
    static void applyThresholds(int[][] matrixOutputs, double[][] matrixOutputsD,
            ArrayList<ArrayList<Double>> datasetTest,
            ArrayList<ArrayList<Integer>> indexesLevels,
            double threshold,
            double thresholdReductionFactor,
            int numLevels) {

        for (int ind = 0; ind < numLevels; ind++) {

            //Index of classes
            int indexFirstClass = indexesLevels.get(ind).get(0);
            int numClassesLast = indexesLevels.get(ind).size();
            int indexLastClass = indexesLevels.get(ind).get(numClassesLast - 1);

            //Iterates of all test instances
            for (int numInst = 0; numInst < datasetTest.size(); numInst++) {

                for (int i = indexFirstClass; i <= indexLastClass; i++) {
                    matrixOutputs[numInst][i] = getOutputThreshold(matrixOutputsD[numInst][i],
                            threshold, thresholdReductionFactor, ind);
                }
            }
        }
    }

    /*===========================================================================
     * Evaluation measure to calculated the Fmeasure per level (Single-Label)
     * Will consider that if the value assign to a class is higher then 0, the
     * class is predicted (1). Will not penalize over-specialized predictions
     *===========================================================================*/
    public static double[] evaluationFmeasure(double[][] matrixPredictions,
            ArrayList<ArrayList<Double>> datasetTest,
            ArrayList<ArrayList<Integer>> indexesLevels,
            double threshold,
            double thresholdReductionFactor,
            int numLevels,
            int multilabel) {


        double[] fmeasureLevels = new double[numLevels];

        //Matrix to store the outputs on the test data
        int[][] binaryMatrix = new int[matrixPredictions.length][matrixPredictions[0].length];

        applyThresholds(binaryMatrix, matrixPredictions, datasetTest,
                indexesLevels, threshold, thresholdReductionFactor, numLevels);

        for (int ind = 0; ind < numLevels; ind++) {

            int indexFirstClass = indexesLevels.get(ind).get(0);
            int numClassesLast = indexesLevels.get(ind).size();
            int indexLastClass = indexesLevels.get(ind).get(numClassesLast - 1);

            //F-measure per level
            double fmeasure = evaluationFmeasureLevel(datasetTest, binaryMatrix, indexFirstClass, indexLastClass);

            fmeasureLevels[ind] = fmeasure;
        }

        return fmeasureLevels;
    }
    
    /* ===========================================================
     * Save the fmeasure per level (Single-Label version)
     * =========================================================== */
    public static void saveFmeasureRun(double[] fmeasureLevels, int errChoice, int learningAlgorithm,
            String nameDatasetTest, int numRun, int numLevels) {

        try {

            Paths paths = new Paths();
            String dirErr = paths.setErrorChoice(errChoice);
            String dirAlg = paths.setLearningAlgorithm(learningAlgorithm);

            paths.setPaths(dirAlg, dirErr, nameDatasetTest, numRun);

            String fileFmeasureLevels = paths.getFileFmeasureLevels();

            //Results in each level
            File fmeasures = new File(fileFmeasureLevels);
            FileWriter fstream = new FileWriter(fmeasures);
            BufferedWriter out = new BufferedWriter(fstream);
            int numLevel = 0;

            for (int level = 0; level < numLevels; level++) {
                numLevel++;
                out.write("Fmeasure level " + numLevel + " = " + fmeasureLevels[level] + "\n");
            }

            out.close();

        } catch (IOException ioe) {
            ioe.printStackTrace();
        }
    }

    /*===========================================================================
     * Calculate the F-measure for a level, given an array list with the 
     * position of the classes for the level
     * Will not penalize over-specialization
     *===========================================================================*/
    static double evaluationFmeasureLevel(ArrayList<ArrayList<Double>> datasetTest, int[][] predictedClasses, int indexFirstClass, int indexLastClass) {

        double fmeasure = 0;
        double sumIntersection = 0;
        double minSumPredicted = 0;
        double sumPredicted = 0;
        double sumReal = 0;
        double FP = 0;
        double FN = 0;

        //Iterates of all test instances
        for (int numInst = 0; numInst < datasetTest.size(); numInst++) {

            double sumPredictedExample = 0;
            double sumRealExample = 0;

            for (int i = indexFirstClass; i <= indexLastClass; i++) {

                if (predictedClasses[numInst][i] == 1 && datasetTest.get(numInst).get(i) == 1) {
                    sumIntersection++;
                }
                if (predictedClasses[numInst][i] == 1) {
                    sumPredictedExample++;
                    sumPredicted++;
                }
                if (datasetTest.get(numInst).get(i) == 1) {
                    sumRealExample++;
                    sumReal++;
                }
                if (predictedClasses[numInst][i] == 1 && datasetTest.get(numInst).get(i) != 1) {
                    FP++;
                }
                if (predictedClasses[numInst][i] != 1 && datasetTest.get(numInst).get(i) == 1) {
                    FN++;
                }
            }

            //Get the minimum value. This will not penalize over-specialization
            if (sumPredictedExample < sumRealExample) {
                minSumPredicted += sumPredictedExample;
            } else {
                minSumPredicted += sumRealExample;
            }
        }

        //Hierarchical Precision
        double hPrecision = 0.0;
        if (minSumPredicted != 0) {
            hPrecision = sumIntersection / minSumPredicted;
        }

        //Hierarchical Recall
        double hRecall = 0.0;
        if (sumReal != 0) {
            hRecall = sumIntersection / sumReal;
        }

        //Fmeasure
        if (hPrecision != 0 || hRecall != 0) {
            fmeasure = (2 * hPrecision * hRecall) / (hPrecision + hRecall);
        }

        return fmeasure;
    }

    /*===========================================================================
     * Save a confusion matrix
     *===========================================================================*/
    static void saveConfusionMatrices(String nameDatasetTest,
            int[][] confusionMatrices,
            int errChoice, int numRun,
            int learningAlgorithm, int numberEpochs) {
        Paths paths = new Paths();

        String dirErr = paths.setErrorChoice(errChoice);
        String dirAlg = paths.setLearningAlgorithm(learningAlgorithm);

        paths.setPathsSingleLabel(dirAlg, dirErr, numberEpochs, nameDatasetTest, numRun);

        try {
            FileOutputStream fileStream = new FileOutputStream(paths.getFileConfusionMatrices());
            ObjectOutputStream os = new ObjectOutputStream(fileStream);
            os.writeObject(confusionMatrices);
            os.close();
        } catch (Exception ex) {
            ex.printStackTrace();
        }

    }

    /*===========================================================================
     * Create directories for output results
     *===========================================================================*/
    static void createDirectories(String nameDatasetValid, String nameDatasetTest,
            ArrayList<Integer> numEpochs,
            int errChoice,
            int numRuns,
            int learningAlgorithm) {

        Paths paths = new Paths();

        String dirErr = paths.setErrorChoice(errChoice);
        String dirAlg = paths.setLearningAlgorithm(learningAlgorithm);

        for (int nRun = 1; nRun <= numRuns; nRun++) {

            paths.setPathsSingleLabel(dirAlg, dirErr, numEpochs, nameDatasetValid, nameDatasetTest, nRun);

            String strDirectoryDatasetValid = paths.getStrDirectoryDatasetValid();

            String strDirectoryDatasetTest = paths.getStrDirectoryDatasetTest();

            new File(strDirectoryDatasetValid).mkdirs();

            new File(strDirectoryDatasetTest).mkdirs();

            for (int i = 0; i < numEpochs.size(); i++) {
                String strDirectoryEpochs = numEpochs.get(i).toString();
                new File(strDirectoryDatasetTest + "/" + strDirectoryEpochs).mkdir();
                new File(strDirectoryDatasetValid + "/" + strDirectoryEpochs).mkdir();
            }
        }
    }

    /*===========================================================================
     * Calculate the output resulted from the first hidden layer and the first level
     *===========================================================================*/
    //Logistic function
    static void outputsFirstHLayerFirstLevel(ArrayList<Integer[]> arrayArchitecture,
            ArrayList<ArrayList<Double[][]>> neuralNet,
            ArrayList<ArrayList<Double[]>> outputs,
            int numInInst, int[][] matrixOutputs, double[][] matrixOutputsD,
            ArrayList<ArrayList<Integer>> indexesLevels, double a,
            ArrayList<ArrayList<ArrayList<Double>>> biasWeights) {

        for (int j = 0; j < (arrayArchitecture.get(0)[1] - arrayArchitecture.get(0)[0]); j++) {
            for (int i = 0; i < outputs.get(0).get(0).length; i++) {
                outputs.get(1).get(0)[j] += outputs.get(0).get(0)[i] * neuralNet.get(1).get(0)[i][j];
            }
            outputs.get(1).get(0)[j] += 1 * biasWeights.get(1).get(0).get(j);
            matrixOutputsD[numInInst][indexesLevels.get(0).get(j)] = FunctionsCommon.sigmoidalLogistic(outputs.get(1).get(0)[j], a);
            outputs.get(1).get(0)[j] = FunctionsCommon.sigmoidalLogistic(outputs.get(1).get(0)[j], a);
            //matrixOutputs[numInInst][indexesLevels.get(0).get(j)] = matrixOutputsD[numInInst][indexesLevels.get(0).get(j)];
        }
        matrixOutputs = getHigherOutput(indexesLevels, arrayArchitecture, numInInst,
                matrixOutputs, matrixOutputsD, 0);
    }

    //Hiperbolic tangent function
    static void outputsFirstHLayerFirstLevel(ArrayList<Integer[]> arrayArchitecture,
            ArrayList<ArrayList<Double[][]>> neuralNet,
            ArrayList<ArrayList<Double[]>> outputs,
            int numInInst, int[][] matrixOutputs, double[][] matrixOutputsD,
            ArrayList<ArrayList<Integer>> indexesLevels, double b, double c,
            ArrayList<ArrayList<ArrayList<Double>>> biasWeights) {

        for (int j = 0; j < (arrayArchitecture.get(0)[1] - arrayArchitecture.get(0)[0]); j++) {
            for (int i = 0; i < outputs.get(0).get(0).length; i++) {
                outputs.get(1).get(0)[j] += outputs.get(0).get(0)[i] * neuralNet.get(1).get(0)[i][j];
            }
            outputs.get(1).get(0)[j] += 1 * biasWeights.get(1).get(0).get(j);
            matrixOutputsD[numInInst][indexesLevels.get(0).get(j)] = FunctionsCommon.sigmoidalHiperbolicTangent(outputs.get(1).get(0)[j], b, c);
            outputs.get(1).get(0)[j] = FunctionsCommon.sigmoidalHiperbolicTangent(outputs.get(1).get(0)[j], b, c);
            //matrixOutputs[numInInst][indexesLevels.get(0).get(j)] = matrixOutputsD[numInInst][indexesLevels.get(0).get(j)];
        }
        matrixOutputs = getHigherOutput(indexesLevels, arrayArchitecture, numInInst,
                matrixOutputs, matrixOutputsD, 0);
    }

    /*===========================================================================
     * Gets the higher output from given outputs - Used in Single-Label version
     *===========================================================================*/
    static int[][] getHigherOutput(ArrayList<ArrayList<Integer>> indexesLevels,
            ArrayList<Integer[]> arrayArchitecture, int numInInst,
            int[][] matrixOutputs, double[][] matrixOutputsD, int actLevel) {

        double higher = 0.0;
        int pos = 0;

        for (int j = 0; j < (arrayArchitecture.get(0)[actLevel + 1] - arrayArchitecture.get(0)[0]); j++) {
            if (higher < matrixOutputsD[numInInst][indexesLevels.get(actLevel).get(j)]) {
                higher = matrixOutputsD[numInInst][indexesLevels.get(actLevel).get(j)];
                pos = indexesLevels.get(actLevel).get(j);
            }
        }

        matrixOutputs[numInInst][pos] = 1;

        return matrixOutputs;
    }

    /*===========================================================================
     * Print time
     *===========================================================================*/
    static void printTime(double time_ms, double time_s, double time_m, double time_h,
            int numberEpochs, int errChoice, String nameDatasetTest,
            String nameOutput, int timeParam, int numRun, int learningAlgorithm) {

        Paths paths = new Paths();

        String dirErr = paths.setErrorChoice(errChoice);
        String dirAlg = paths.setLearningAlgorithm(learningAlgorithm);

        paths.setPathsSingleLabel(dirAlg, dirErr, numberEpochs, nameDatasetTest, numRun);

        String fileOutput = paths.getFilePath() + "/" + nameOutput;

        try {

            File times = new File(fileOutput);
            FileWriter fstream = new FileWriter(times);
            BufferedWriter out = new BufferedWriter(fstream);

            if (timeParam == 1) {
                out.write("Training time\n\n");
            } else {
                out.write("Test time\n\n");
            }
            out.write("Milliseconds = " + time_ms + "\n");
            out.write("Seconds = " + time_s + "\n");
            out.write("Minutes = " + time_m + "\n");
            out.write("Hours = " + time_h + "\n");

            out.close();


        } catch (IOException ioe) {
            ioe.printStackTrace();
        }
    }

    /*===========================================================================
     * Calculate the output resulted from hidden layer and one level
     *===========================================================================*/
    //Logistic function
    static void outputsOneHiddenLayerOneLevel(ArrayList<Integer[]> arrayArchitecture,
            ArrayList<ArrayList<Double[]>> outputs,
            ArrayList<ArrayList<Double[][]>> neuralNet,
            int[][] matrixOutputs, double[][] matrixOutputsD,
            ArrayList<ArrayList<Integer>> indexesLevels,
            int actLevel, double a, int numInInst,
            ArrayList<ArrayList<ArrayList<Double>>> biasWeights) {

        for (int j = 0; j < (arrayArchitecture.get(0)[actLevel + 1] - arrayArchitecture.get(0)[0]); j++) {
            for (int i = 0; i < outputs.get(0).get(actLevel).length; i++) {
                outputs.get(1).get(actLevel)[j] += outputs.get(0).get(actLevel)[i] * neuralNet.get(1).get(actLevel)[i][j];
            }
            outputs.get(1).get(actLevel)[j] += 1 * biasWeights.get(1).get(actLevel).get(j);
            matrixOutputsD[numInInst][indexesLevels.get(actLevel).get(j)] = FunctionsCommon.sigmoidalLogistic(outputs.get(1).get(actLevel)[j], a);
            outputs.get(1).get(actLevel)[j] = FunctionsCommon.sigmoidalLogistic(outputs.get(1).get(actLevel)[j], a);
            //matrixOutputs[numInInst][indexesLevels.get(actLevel).get(j)] = Functions.getOutputThreshold(matrixOutputsD[numInInst][indexesLevels.get(actLevel).get(j)],threshold);
        }
        matrixOutputs = getHigherOutput(indexesLevels, arrayArchitecture, numInInst,
                matrixOutputs, matrixOutputsD, actLevel);
    }

    //Hiperbolic tangent function
    static void outputsOneHiddenLayerOneLevel(ArrayList<Integer[]> arrayArchitecture,
            ArrayList<ArrayList<Double[]>> outputs,
            ArrayList<ArrayList<Double[][]>> neuralNet,
            int[][] matrixOutputs, double[][] matrixOutputsD,
            ArrayList<ArrayList<Integer>> indexesLevels,
            int actLevel, double b, double c, int numInInst,
            ArrayList<ArrayList<ArrayList<Double>>> biasWeights) {

        for (int j = 0; j < (arrayArchitecture.get(0)[actLevel + 1] - arrayArchitecture.get(0)[0]); j++) {
            for (int i = 0; i < outputs.get(0).get(actLevel).length; i++) {
                outputs.get(1).get(actLevel)[j] += outputs.get(0).get(actLevel)[i] * neuralNet.get(1).get(actLevel)[i][j];
            }
            outputs.get(1).get(actLevel)[j] += 1 * biasWeights.get(1).get(actLevel).get(j);
            matrixOutputsD[numInInst][indexesLevels.get(actLevel).get(j)] = FunctionsCommon.sigmoidalHiperbolicTangent(outputs.get(1).get(actLevel)[j], b, c);
            outputs.get(1).get(actLevel)[j] = FunctionsCommon.sigmoidalHiperbolicTangent(outputs.get(1).get(actLevel)[j], b, c);
            //matrixOutputs[numInInst][indexesLevels.get(actLevel).get(j)] = Functions.getOutputThreshold(matrixOutputsD[numInInst][indexesLevels.get(actLevel).get(j)],threshold);
        }
        matrixOutputs = getHigherOutput(indexesLevels, arrayArchitecture, numInInst,
                matrixOutputs, matrixOutputsD, actLevel);
    }

    /*===========================================================================
     * For single-label, lets get only the higher probabilities
     * in the vectors for each level
     *===========================================================================*/
    public static double[][] getHigherProbabilities(double[][] matrixPredictions, int numLevels,
            ArrayList<ArrayList<Integer>> indexesLevels,
            ArrayList<String> namesAttributes) {

        int indexFirstClass = indexesLevels.get(0).get(0);
        int numClassesLast = indexesLevels.get(numLevels - 1).size();
        int indexLastClass = indexesLevels.get(numLevels - 1).get(numClassesLast - 1);

        double[][] matrixPredictionsSingleLabel = new double[matrixPredictions.length][matrixPredictions[0].length];

        for (int i = 0; i < matrixPredictions.length; i++) {

            String patternClass = "";

            for (int level = 0; level < numLevels; level++) {

                Pattern pattern = Pattern.compile("^" + patternClass + "[0-9]+$");

                double higher = 0;
                int indexHigher = 0;
                for (int indexClass = indexFirstClass; indexClass <= indexLastClass; indexClass++) {

                    Matcher m = pattern.matcher(namesAttributes.get(indexClass));
                    if (m.find()) {

                        if (matrixPredictions[i][indexClass] > higher) {
                            higher = matrixPredictions[i][indexClass];
                            indexHigher = indexClass;
                        }
                    }
                }

                if (higher > 0) {
                    matrixPredictionsSingleLabel[i][indexHigher] = higher;
                    patternClass = namesAttributes.get(indexHigher) + "\\.";
                } else {
                    break;
                }
            }
        }

        return matrixPredictionsSingleLabel;
    }


    /*===========================================================================
     * Verify if there is a class predicted in the current level
     *===========================================================================*/
    static int verifyIsPredicted(int[][] matrixOutputs, int numInst,
            int indexFirstClass, int indexLastClass) {

        int isPredicted = 0;

        for (int i = indexFirstClass; i <= indexLastClass; i++) {
            if (matrixOutputs[numInst][i] == 1) {
                isPredicted = 1;
                break;
            }
        }

        return isPredicted;
    }

    /*===========================================================================
     * Recover the deepest predicted class for an example
     *===========================================================================*/
    static String recoverLastPrediction(ArrayList<String> namesAttributes,
            int[][] matrixOutputs,
            ArrayList<ArrayList<Integer>> indexesLevels,
            int level,
            int numInst) {

        String lastPrediction = "";

        int indexFirstClass = indexesLevels.get(level).get(0);
        int numClassesLast = indexesLevels.get(level).size();
        int indexLastClass = indexesLevels.get(level).get(numClassesLast - 1);

        for (int i = indexFirstClass; i <= indexLastClass; i++) {
            if (matrixOutputs[numInst][i] == 1) {
                lastPrediction = namesAttributes.get(i);
                break;
            }
        }

        return lastPrediction;
    }

    /*===========================================================================
     * Generate random predictions for levels where no predictions were make
     * Used in Mandatory Leaf Node Classification
     *===========================================================================*/
    static int generateRandomPredictions(ArrayList<ArrayList<Integer>> indexesLevels,
            ArrayList<String> namesAttributes,
            int[][] matrixOutputs,
            int level,
            String lastPrediction,
            int numLevels,
            int numInst) {
        int pos = -1;

        //Putz the class in a regular expression format
        String currentClass = FunctionsCommon.putClassRegExpFormat(lastPrediction);

        for (int ind = level; ind < numLevels; ind++) {

            String regExp = "^" + currentClass + "\\.";

            Pattern pattern = Pattern.compile(regExp);

            int indexFirstClass = indexesLevels.get(ind).get(0);
            int numClassesLast = indexesLevels.get(ind).size();
            int indexLastClass = indexesLevels.get(ind).get(numClassesLast - 1);

            ArrayList<Integer> subClasses = new ArrayList<Integer>();

            //Get the subclasses of the current class
            //There can be that no subclass is found because there is not a next level
            //In this case, subClasses will be empty
            //This situation will occur in unbalanced hierarchies, in which
            //there are classes in the first level that are leaf classes
            //of the hierarchy
            for (int j = indexFirstClass; j < indexLastClass; j++) {
                Matcher m = pattern.matcher(namesAttributes.get(j));
                if (m.find()) {
                    subClasses.add(j);
                }
            }

            //If subClasses not higher than 0, there is no next level, 
            //and pos will be equal -1
            if (subClasses.size() > 0) {
                //Randomly choose one of the subclasses
                Random randomGenerator = new Random();
                int numSubClasses = subClasses.size();
                int randClass = randomGenerator.nextInt(numSubClasses);
                matrixOutputs[numInst][subClasses.get(randClass)] = 1;

                //Position of the choosen class. Must be returned
                pos = subClasses.get(randClass);

                //Get the chosen class in order to
                //search for classes in the next level
                currentClass = FunctionsCommon.putClassRegExpFormat(namesAttributes.get(subClasses.get(randClass)));
            }
        }

        return pos;
    }

    //Gets a random class class for the first level
    static int generateRandomPredictions(ArrayList<ArrayList<Integer>> indexesLevels,
            ArrayList<String> namesAttributes,
            int[][] matrixOutputs,
            int level,
            int numLevels,
            int numInst) {
        int pos = 0;

        //Index of classes
        int indexFirstClass = indexesLevels.get(0).get(0);
        int numClassesLast = indexesLevels.get(0).size();
        int indexLastClass = indexesLevels.get(0).get(numClassesLast - 1);

        //No class was predicted in any level.
        //So, first randomly choose a class for the first level
        ArrayList<Integer> classes = new ArrayList<Integer>();
        for (int i = indexFirstClass; i < indexLastClass; i++) {
            classes.add(i);
        }

        //Randomly choose one of the subclasses
        Random randomGenerator = new Random();
        int numClasses = classes.size();
        int randClass = randomGenerator.nextInt(numClasses);
        matrixOutputs[numInst][classes.get(randClass)] = 1;

        //Get the chosen class
        String currentClass = namesAttributes.get(classes.get(randClass));

        //Choose the other classes and get the position of
        //the class in the last level
        pos = generateRandomPredictions(indexesLevels, namesAttributes, matrixOutputs,
                level + 1, currentClass, numLevels, numInst);

        return pos;
    }

    /*===========================================================================
     * Get the position of the predicted class to construct confusion matrices
     * Mandatory leaf node classification version
     *===========================================================================*/
    static int getPosPredictedClassMLNC(int[][] matrixOutputs, int numInst,
            int indexFirstClass, int indexLastClass,
            ArrayList<String> namesAttributes,
            ArrayList<ArrayList<Integer>> indexesLevels,
            int numLevels) {

        int pos = 0;

        //Verify what class was predicted
        for (int i = indexFirstClass; i <= indexLastClass; i++) {

            if (matrixOutputs[numInst][i] == 1) {
                pos = i;
                break;
            }
        }

        //If pos == 0, there was no prediction in the last level, so 
        //a random subclass is chosen
        if (pos == 0) {

            //Verify what was the last level where a class was predicted
            //From this level onwards, random subclasses will be chosen
            int level;
            String lastPrediction;
            for (level = 0; level < numLevels; level++) {

                indexFirstClass = indexesLevels.get(level).get(0);
                int numClassesLast = indexesLevels.get(level).size();
                indexLastClass = indexesLevels.get(level).get(numClassesLast - 1);

                //Verify if there is a predicte class in this level
                int isPredicted = verifyIsPredicted(matrixOutputs, numInst,
                        indexFirstClass, indexLastClass);

                //There was no predicted class in this level.
                //Recover the predicted class of the previous level
                //Must also check if the current level is the first. If so, it's
                //impossible to check the previous class, and a ramdom class will
                //be chosen for the first class
                if (isPredicted == 0 && level > 0) {
                    lastPrediction = recoverLastPrediction(namesAttributes, matrixOutputs,
                            indexesLevels, level - 1, numInst);

                    //Now that we have the levels where random classes should be choosen,
                    //and the last prediction, choose classes at random
                    pos = generateRandomPredictions(indexesLevels, namesAttributes, matrixOutputs,
                            level, lastPrediction, numLevels, numInst);
                    break;
                }
                if (isPredicted == 0 && level == 0) {
                    //No classes were predicted in any levels
                    //In this case, generate a complete path in the hierarchy randomly
                    pos = generateRandomPredictions(indexesLevels, namesAttributes,
                            matrixOutputs, level, numLevels, numInst);
                    break;
                }
            }//END for(level)
        }//END pos==0

        return pos;
    }

    /*===========================================================================
     * Calculate the accuracy only considering the leaf classes of the hierarchy
     *===========================================================================*/
    static double calculateAccuracyLeaves(ArrayList<ArrayList<Integer>> indexesLevels,
            ArrayList<String> namesAttributes,
            int[][] confusionMatrices) {

        int correctPredictions = 0;
        double accuracy = 0.0;
        int indexFirstClass = indexesLevels.get(0).get(0);
        int countTotal = 0;

        //For each class, verify if it is a leaf class
        for (int i = indexFirstClass; i < namesAttributes.size(); i++) {
            int isLeaf = FunctionsCommon.verifyIsLeaf(namesAttributes.get(i),
                    namesAttributes, indexFirstClass);

            //If the class is a leaf, get its position
            if (isLeaf == 1) {
                //String regExpClass = FunctionsCommon.putClassRegExpFormat(namesAttributes.get(i));
                //int pos = FunctionsCommon.getPosSpecificClass(regExpClass, namesAttributes, indexFirstClass);

                for (int j = indexFirstClass; j < namesAttributes.size(); j++) {
                    countTotal += confusionMatrices[i - indexFirstClass][j - indexFirstClass];
                }

                //Count the number of correct predictions
                correctPredictions += confusionMatrices[i - indexFirstClass][i - indexFirstClass];
            }
        }

        //Calculate the accuracy
        accuracy = (double) correctPredictions / (double) countTotal;

        return accuracy;
    }

    /*===========================================================================
     * Weighted True positive rate for a specific level
     * Do TP/(TP+FN) for each class. Then sum this weighted by the number
     * of examples belonging to the class and divide by the number of examples
     * Also called Recalll
     *===========================================================================*/
    static double calculateWeightedRecallLevel(int[][] confusionMatrices,
            int indexFirstClass,
            int indexFirstClassCurrent,
            int indexLastClass) {

        double truePositiveRate = 0.0;
        int totalCount = 0;

        for (int i = indexFirstClassCurrent; i <= indexLastClass; i++) {

            int truePositives = 0;
            int falseNegatives = 0;
            double truePositiveRateClass = 0.0;


            for (int j = indexFirstClassCurrent; j <= indexLastClass; j++) {

                if (i == j) {
                    truePositives = truePositives + confusionMatrices[i - indexFirstClass][j - indexFirstClass];
                } else {
                    falseNegatives = falseNegatives + confusionMatrices[i - indexFirstClass][j - indexFirstClass];
                }
            }

            totalCount = totalCount + truePositives + falseNegatives;

            //Still thinking, but there can be the case where truePositives + falseNegatives == 0
            //in a non-mandatory leaf node classification 
            if ((truePositives + falseNegatives) > 0) {
                truePositiveRateClass = ((double) truePositives / (double) (truePositives + falseNegatives)) * (double) (truePositives + falseNegatives);
            }
            truePositiveRate = truePositiveRate + truePositiveRateClass;
        }

        truePositiveRate = truePositiveRate / totalCount;

        return truePositiveRate;
    }

    /*===========================================================================
     * Calculate the weighted true positive rate only considering
     * the leaf classes of the hierarchy
     * Also called Recalll
     *===========================================================================*/
    static double calculateWeightedRecallLeaves(ArrayList<ArrayList<Integer>> indexesLevels,
            ArrayList<String> namesAttributes,
            int[][] confusionMatrices) {

        double truePositiveRate = 0.0;
        int indexFirstClass = indexesLevels.get(0).get(0);
        int totalCount = 0;

        //For each class, verify if it is a leaf class
        for (int i = indexFirstClass; i < namesAttributes.size(); i++) {
            int isLeaf = FunctionsCommon.verifyIsLeaf(namesAttributes.get(i),
                    namesAttributes, indexFirstClass);

            //If the class is a leaf, get its position
            if (isLeaf == 1) {
                //String regExpClass = FunctionsCommon.putClassRegExpFormat(namesAttributes.get(i));
                //int pos = FunctionsCommon.getPosSpecificClass(regExpClass, namesAttributes, indexFirstClass);

                int truePositives = 0;
                int falseNegatives = 0;
                double truePositiveRateClass = 0.0;

                for (int j = indexFirstClass; j < namesAttributes.size(); j++) {

                    if (i == j) {
                        truePositives = truePositives + confusionMatrices[i - indexFirstClass][j - indexFirstClass];
                    } else {
                        falseNegatives = falseNegatives + confusionMatrices[i - indexFirstClass][j - indexFirstClass];
                    }
                }

                totalCount = totalCount + truePositives + falseNegatives;

                if ((truePositives + falseNegatives) > 0) {
                    truePositiveRateClass = ((double) truePositives / (double) (truePositives + falseNegatives)) * (double) (truePositives + falseNegatives);
                }
                truePositiveRate = truePositiveRate + truePositiveRateClass;
            }
        }

        //Calculate the true positive rate
        truePositiveRate = (double) truePositiveRate / (double) totalCount;

        return truePositiveRate;
    }

    /*===========================================================================
     * Weighted precision for a specific level
     * Do TP/(TP+FP) for each class. Then sum this weighted by the number
     * of examples belonging to the class and divide by the number of examples
     *===========================================================================*/
    static double calculateWeightedPrecisionLevel(int[][] confusionMatrices,
            int indexFirstClass,
            int indexFirstClassCurrent,
            int indexLastClass) {

        double precision = 0.0;
        int totalCount = 0;

        for (int i = indexFirstClassCurrent; i <= indexLastClass; i++) {

            int truePositives = 0;
            int falseNegatives = 0;
            int falsePositives = 0;
            double precisionClass = 0.0;

            for (int j = indexFirstClassCurrent; j <= indexLastClass; j++) {

                if (i == j) {
                    truePositives = truePositives + confusionMatrices[j - indexFirstClass][i - indexFirstClass];
                } else {
                    falsePositives = falsePositives + confusionMatrices[j - indexFirstClass][i - indexFirstClass];
                    falseNegatives = falseNegatives + confusionMatrices[i - indexFirstClass][j - indexFirstClass];
                }
            }

            totalCount = totalCount + truePositives + falseNegatives;

            if ((truePositives + falsePositives) > 0) {
                precisionClass = ((double) truePositives / (double) (truePositives + falsePositives)) * (double) (truePositives + falseNegatives);
            }
            precision = precision + precisionClass;
        }

        precision = precision / totalCount;

        return precision;
    }

    /*===========================================================================
     * Weighted precision for all leaf classes
     * Do TP/(TP+FP) for each class. Then sum this weighted by the number
     * of examples belonging to the class and divide by the number of examples
     *===========================================================================*/
    static double calculateWeightedPrecisionLeaves(ArrayList<ArrayList<Integer>> indexesLevels,
            ArrayList<String> namesAttributes,
            int[][] confusionMatrices) {

        double precision = 0.0;
        int indexFirstClass = indexesLevels.get(0).get(0);
        int totalCount = 0;

        //For each class, verify if it is a leaf class
        for (int i = indexFirstClass; i < namesAttributes.size(); i++) {
            int isLeaf = FunctionsCommon.verifyIsLeaf(namesAttributes.get(i),
                    namesAttributes, indexFirstClass);

            //If the class is a leaf, get its position
            if (isLeaf == 1) {
                //String regExpClass = FunctionsCommon.putClassRegExpFormat(namesAttributes.get(i));
                //int pos = FunctionsCommon.getPosSpecificClass(regExpClass, namesAttributes, indexFirstClass);

                int truePositives = 0;
                int falseNegatives = 0;
                int falsePositives = 0;
                double precisionClass = 0.0;

                for (int j = indexFirstClass; j < namesAttributes.size(); j++) {

                    if (i == j) {
                        truePositives = truePositives + confusionMatrices[i - indexFirstClass][j - indexFirstClass];
                    } else {
                        falsePositives = falsePositives + confusionMatrices[j - indexFirstClass][i - indexFirstClass];
                        falseNegatives = falseNegatives + confusionMatrices[i - indexFirstClass][j - indexFirstClass];
                    }
                }

                totalCount = totalCount + truePositives + falseNegatives;

                if ((truePositives + falsePositives) > 0) {
                    precisionClass = ((double) truePositives / (double) (truePositives + falsePositives)) * (double) (truePositives + falseNegatives);
                }
                precision = precision + precisionClass;
            }
        }

        //Calculate the precision
        precision = (double) precision / (double) totalCount;

        return precision;
    }

    /*===========================================================================
     * Weighted f-measure for all leaf classes
     *===========================================================================*/
    static double calculateWeightedFmeasureLeaves(ArrayList<ArrayList<Integer>> indexesLevels,
            ArrayList<String> namesAttributes,
            int[][] confusionMatrices) {

        double fmeasure = 0.0;
        int indexFirstClass = indexesLevels.get(0).get(0);
        int totalCount = 0;

        //For each class, verify if it is a leaf class
        for (int i = indexFirstClass; i < namesAttributes.size(); i++) {
            int isLeaf = FunctionsCommon.verifyIsLeaf(namesAttributes.get(i),
                    namesAttributes, indexFirstClass);

            //If the class is a leaf, get its position
            if (isLeaf == 1) {
                //String regExpClass = FunctionsCommon.putClassRegExpFormat(namesAttributes.get(i));
                //int pos = FunctionsCommon.getPosSpecificClass(regExpClass, namesAttributes, indexFirstClass);

                int truePositives = 0;
                int falseNegatives = 0;
                int falsePositives = 0;
                double precisionClass = 0.0;
                double recallClass = 0.0;

                for (int j = indexFirstClass; j < namesAttributes.size(); j++) {

                    if (i == j) {
                        truePositives = truePositives + confusionMatrices[i - indexFirstClass][j - indexFirstClass];
                    } else {
                        falsePositives = falsePositives + confusionMatrices[j - indexFirstClass][i - indexFirstClass];
                        falseNegatives = falseNegatives + confusionMatrices[i - indexFirstClass][j - indexFirstClass];
                    }
                }

                totalCount = totalCount + truePositives + falseNegatives;

                if ((truePositives + falsePositives) > 0) {
                    precisionClass = (double) truePositives / (double) (truePositives + falsePositives);
                }
                if ((truePositives + falseNegatives) > 0) {
                    recallClass = (double) truePositives / (double) (truePositives + falseNegatives);
                }

                if ((precisionClass + recallClass) > 0) {
                    fmeasure = fmeasure + ((2 * precisionClass * recallClass) / (precisionClass + recallClass)) * (double) (truePositives + falseNegatives);
                }
            }
        }

        fmeasure = fmeasure / (double) totalCount;

        return fmeasure;
    }

    /*===========================================================================
     * Weighted Unweighted f-measure for all leaf classes
     *===========================================================================*/
    static double calculateUnweightedFmeasureLeaves(ArrayList<ArrayList<Integer>> indexesLevels,
            ArrayList<String> namesAttributes,
            int[][] confusionMatrices) {

        double fmeasure = 0.0;
        int indexFirstClass = indexesLevels.get(0).get(0);
        int totalCount = 0;

        //For each class, verify if it is a leaf class
        for (int i = indexFirstClass; i < namesAttributes.size(); i++) {
            int isLeaf = FunctionsCommon.verifyIsLeaf(namesAttributes.get(i),
                    namesAttributes, indexFirstClass);

            //If the class is a leaf, get its position
            if (isLeaf == 1) {
                //String regExpClass = FunctionsCommon.putClassRegExpFormat(namesAttributes.get(i));
                //int pos = FunctionsCommon.getPosSpecificClass(regExpClass, namesAttributes, indexFirstClass);

                int truePositives = 0;
                int falseNegatives = 0;
                int falsePositives = 0;
                double precisionClass = 0.0;
                double recallClass = 0.0;

                for (int j = indexFirstClass; j < namesAttributes.size(); j++) {

                    if (i == j) {
                        truePositives = truePositives + confusionMatrices[i - indexFirstClass][j - indexFirstClass];
                    } else {
                        falsePositives = falsePositives + confusionMatrices[j - indexFirstClass][i - indexFirstClass];
                        falseNegatives = falseNegatives + confusionMatrices[i - indexFirstClass][j - indexFirstClass];
                    }
                }

                if ((truePositives + falsePositives) > 0) {
                    precisionClass = (double) truePositives / (double) (truePositives + falsePositives);
                }
                if ((truePositives + falseNegatives) > 0) {
                    recallClass = (double) truePositives / (double) (truePositives + falseNegatives);
                    totalCount++;
                }

                if ((precisionClass + recallClass) > 0) {
                    fmeasure = fmeasure + (2 * precisionClass * recallClass) / (precisionClass + recallClass);
                }
            }
        }

        fmeasure = fmeasure / (double) totalCount;

        return fmeasure;
    }

    /*===========================================================================
     * Weighted f-measure for a specific level
     *===========================================================================*/
    static double calculateWeightedFmeasureLevel(int[][] confusionMatrices,
            int indexFirstClass,
            int indexFirstClassCurrent,
            int indexLastClass) {

        double fmeasure = 0.0;
        int totalCount = 0;

        for (int i = indexFirstClassCurrent; i <= indexLastClass; i++) {

            int truePositives = 0;
            int falseNegatives = 0;
            int falsePositives = 0;
            double precisionClass = 0.0;
            double recallClass = 0.0;

            for (int j = indexFirstClassCurrent; j <= indexLastClass; j++) {

                if (i == j) {
                    truePositives = truePositives + confusionMatrices[j - indexFirstClass][i - indexFirstClass];
                } else {
                    falsePositives = falsePositives + confusionMatrices[j - indexFirstClass][i - indexFirstClass];
                    falseNegatives = falseNegatives + confusionMatrices[i - indexFirstClass][j - indexFirstClass];
                }
            }

            totalCount = totalCount + truePositives + falseNegatives;

            if ((truePositives + falsePositives) > 0) {
                precisionClass = (double) truePositives / (double) (truePositives + falsePositives);
            }
            if ((truePositives + falseNegatives) > 0) {
                recallClass = (double) truePositives / (double) (truePositives + falseNegatives);
            }

            if ((precisionClass + recallClass) > 0) {
                fmeasure = fmeasure + ((2 * precisionClass * recallClass) / (precisionClass + recallClass)) * (double) (truePositives + falseNegatives);
            }
        }

        fmeasure = fmeasure / (double) totalCount;

        return fmeasure;
    }

    /*===========================================================================
     * Unweighted f-measure for a specific level
     *===========================================================================*/
    static double calculateUnweightedFmeasureLevel(int[][] confusionMatrices,
            int indexFirstClass,
            int indexFirstClassCurrent,
            int indexLastClass) {

        double fmeasure = 0.0;
        int totalCount = 0;

        for (int i = indexFirstClassCurrent; i <= indexLastClass; i++) {

            int truePositives = 0;
            int falseNegatives = 0;
            int falsePositives = 0;
            double precisionClass = 0.0;
            double recallClass = 0.0;

            for (int j = indexFirstClassCurrent; j <= indexLastClass; j++) {

                if (i == j) {
                    truePositives = truePositives + confusionMatrices[j - indexFirstClass][i - indexFirstClass];
                } else {
                    falsePositives = falsePositives + confusionMatrices[j - indexFirstClass][i - indexFirstClass];
                    falseNegatives = falseNegatives + confusionMatrices[i - indexFirstClass][j - indexFirstClass];
                }
            }

            if ((truePositives + falsePositives) > 0) {
                precisionClass = (double) truePositives / (double) (truePositives + falsePositives);
            }
            if ((truePositives + falseNegatives) > 0) {
                totalCount++;
                recallClass = (double) truePositives / (double) (truePositives + falseNegatives);
            }

            if ((precisionClass + recallClass) > 0) {
                fmeasure = fmeasure + (2 * precisionClass * recallClass) / (precisionClass + recallClass);
            }
        }

        fmeasure = fmeasure / (double) totalCount;

        return fmeasure;
    }

    /*===========================================================================
     * Unweighted true positive rate for a specific level
     * Do TP/(TP+FN) for each class. Then sum this and divide
     * by the number of classes
     * Also called Recalll
     *===========================================================================*/
    static double calculateUnweightedRecallLevel(int[][] confusionMatrices,
            int indexFirstClass,
            int indexFirstClassCurrent,
            int indexLastClass) {

        double truePositiveRate = 0.0;
        int totalCount = 0;

        for (int i = indexFirstClassCurrent; i <= indexLastClass; i++) {

            int truePositives = 0;
            int falseNegatives = 0;
            double truePositiveRateClass = 0.0;


            for (int j = indexFirstClassCurrent; j <= indexLastClass; j++) {

                if (i == j) {
                    truePositives = truePositives + confusionMatrices[i - indexFirstClass][j - indexFirstClass];
                } else {
                    falseNegatives = falseNegatives + confusionMatrices[i - indexFirstClass][j - indexFirstClass];
                }
            }

            //Still thinking, but there can be the case where truePositives + falseNegatives == 0
            //in a non-mandatory leaf node classification 
            if ((truePositives + falseNegatives) > 0) {
                truePositiveRateClass = (double) truePositives / (double) (truePositives + falseNegatives);
                totalCount++;
            }
            truePositiveRate = truePositiveRate + truePositiveRateClass;
        }

        truePositiveRate = truePositiveRate / totalCount;

        return truePositiveRate;
    }

    /*===========================================================================
     * Calculate the unweighted true positive rate only considering
     * the leaf classes of the hierarchy
     * Also called Recalll
     *===========================================================================*/
    static double calculateUnweightedRecallLeaves(ArrayList<ArrayList<Integer>> indexesLevels,
            ArrayList<String> namesAttributes,
            int[][] confusionMatrices) {

        double truePositiveRate = 0.0;
        int indexFirstClass = indexesLevels.get(0).get(0);
        int totalCount = 0;

        //For each class, verify if it is a leaf class
        for (int i = indexFirstClass; i < namesAttributes.size(); i++) {
            int isLeaf = FunctionsCommon.verifyIsLeaf(namesAttributes.get(i),
                    namesAttributes, indexFirstClass);

            //If the class is a leaf, get its position
            if (isLeaf == 1) {
                //String regExpClass = FunctionsCommon.putClassRegExpFormat(namesAttributes.get(i));
                //int pos = FunctionsCommon.getPosSpecificClass(regExpClass, namesAttributes, indexFirstClass);

                int truePositives = 0;
                int falseNegatives = 0;
                double truePositiveRateClass = 0.0;

                for (int j = indexFirstClass; j < namesAttributes.size(); j++) {

                    if (i == j) {
                        truePositives = truePositives + confusionMatrices[i - indexFirstClass][j - indexFirstClass];
                    } else {
                        falseNegatives = falseNegatives + confusionMatrices[i - indexFirstClass][j - indexFirstClass];
                    }
                }

                if ((truePositives + falseNegatives) > 0) {
                    truePositiveRateClass = (double) truePositives / (double) (truePositives + falseNegatives);
                    totalCount++;
                }
                truePositiveRate = truePositiveRate + truePositiveRateClass;
            }
        }

        //Calculate the true positive rate
        truePositiveRate = (double) truePositiveRate / (double) totalCount;

        return truePositiveRate;
    }

    /*===========================================================================
     * Unweighted precision for a specific level
     * Do TP/(TP+FP) for each class. Then sum this
     * and divide by the number of classes
     *===========================================================================*/
    static double calculateUnweightedPrecisionLevel(int[][] confusionMatrices,
            int indexFirstClass,
            int indexFirstClassCurrent,
            int indexLastClass) {

        double precision = 0.0;
        int totalCount = 0;

        for (int i = indexFirstClassCurrent; i <= indexLastClass; i++) {

            int truePositives = 0;
            int falsePositives = 0;
            int falseNegatives = 0;
            double precisionClass = 0.0;

            for (int j = indexFirstClassCurrent; j <= indexLastClass; j++) {

                if (i == j) {
                    truePositives = truePositives + confusionMatrices[j - indexFirstClass][i - indexFirstClass];
                } else {
                    falsePositives = falsePositives + confusionMatrices[j - indexFirstClass][i - indexFirstClass];
                    falseNegatives = falseNegatives + confusionMatrices[i - indexFirstClass][j - indexFirstClass];
                }
            }

            if ((truePositives + falseNegatives) > 0) {
                totalCount++;
            }

            if ((truePositives + falsePositives) > 0) {
                precisionClass = (double) truePositives / (double) (truePositives + falsePositives);
            }
            precision = precision + precisionClass;
        }

        precision = precision / totalCount;

        return precision;
    }

    /*===========================================================================
     * Unweighted precision for all leaf classes
     * Do TP/(TP+FP) for each class. Then sum this
     * and divide by the number of classes
     *===========================================================================*/
    static double calculateUnweightedPrecisionLeaves(ArrayList<ArrayList<Integer>> indexesLevels,
            ArrayList<String> namesAttributes,
            int[][] confusionMatrices) {

        double precision = 0.0;
        int indexFirstClass = indexesLevels.get(0).get(0);
        int totalCount = 0;

        //For each class, verify if it is a leaf class
        for (int i = indexFirstClass; i < namesAttributes.size(); i++) {
            int isLeaf = FunctionsCommon.verifyIsLeaf(namesAttributes.get(i),
                    namesAttributes, indexFirstClass);

            //If the class is a leaf, get its position
            if (isLeaf == 1) {
                //String regExpClass = FunctionsCommon.putClassRegExpFormat(namesAttributes.get(i));
                //int pos = FunctionsCommon.getPosSpecificClass(regExpClass, namesAttributes, indexFirstClass);

                int truePositives = 0;
                int falsePositives = 0;
                int falseNegatives = 0;
                double precisionClass = 0.0;

                for (int j = indexFirstClass; j < namesAttributes.size(); j++) {

                    if (i == j) {
                        truePositives = truePositives + confusionMatrices[i - indexFirstClass][j - indexFirstClass];
                    } else {
                        falsePositives = falsePositives + confusionMatrices[j - indexFirstClass][i - indexFirstClass];
                        falseNegatives = falseNegatives + confusionMatrices[i - indexFirstClass][j - indexFirstClass];
                    }
                }

                if ((truePositives + falseNegatives) > 0) {
                    totalCount++;
                }

                if ((truePositives + falsePositives) > 0) {
                    precisionClass = (double) truePositives / (double) (truePositives + falsePositives);
                }
                precision = precision + precisionClass;
            }
        }

        //Calculate the precision
        precision = (double) precision / (double) totalCount;

        return precision;
    }

    /*===========================================================================
     * Precision, Recall, F-Measure and Accuracy evaluation metrics
     * Single-Label, so uses confusion matrices to calculate the measures
     *===========================================================================*/
    static double[][] evaluationPrecRecAcc(String nameDatasetTest, ArrayList<ArrayList<Double>> datasetTest,
            int[][] matrixOutputs, ArrayList<ArrayList<Integer>> indexesLevels,
            int numLevels, ArrayList<String> namesAttributes,
            int errChoice, int numRun, int learningAlgorithm, int numberEpochs) {

        //Store the results
        double[][] evalResults = new double[12][numLevels];

        //Index of classes
        int indexFirstClass = indexesLevels.get(0).get(0);
        int numClassesLast = indexesLevels.get(numLevels - 1).size();
        int indexLastClass = indexesLevels.get(numLevels - 1).get(numClassesLast - 1);
        int numClasses = (indexLastClass - indexFirstClass) + 1;

        //Structure to store confusion matrices for each level
        //Just one matrix to store all confusion matrices
        int[][] confusionMatrices = new int[numClasses][numClasses];

        //Go over all levels
        for (int ind = (numLevels - 1); ind >= 0; ind--) {

            //Index of classes of the current level
            int indexFirstClassCurrent = indexesLevels.get(ind).get(0);
            numClassesLast = indexesLevels.get(ind).size();
            indexLastClass = indexesLevels.get(ind).get(numClassesLast - 1);

            //Number of correct and wrong predictions in current level
            int correctPredictionsLevel = 0;
            int wrongPredictionsLevel = 0;

            //Iterate over all classes to construct confusion matrix for the current level
            for (int i = indexFirstClassCurrent; i <= indexLastClass; i++) {

                //Iterates over all test instances
                for (int numInst = 0; numInst < datasetTest.size(); numInst++) {

                    //Correct prediction
                    if (datasetTest.get(numInst).get(i) == 1 && matrixOutputs[numInst][i] == 1) {
                        confusionMatrices[i - indexFirstClass][i - indexFirstClass]++;
                        correctPredictionsLevel++;
                    }

                    //Wrong prediction
                    if (datasetTest.get(numInst).get(i) == 1 && matrixOutputs[numInst][i] == 0) {

                        //Wrong prediction. Need to check what class was predicted
                        //This function makes sure a class was predicted in the last level
                        //This is a Mandatory Leaf Node Classification (MLNC)
                        int posPredictedClass = getPosPredictedClassMLNC(matrixOutputs, numInst, indexFirstClassCurrent,
                                indexLastClass, namesAttributes, indexesLevels,
                                numLevels);

                        //There can be that no subclass is found because there is not a next level
                        //This situation will occur in unbalanced hierarchies, in which
                        //there are classes in the first level that are leaf classes
                        //of the hierarchy
                        //In this case, pos will be equal -1
                        if (posPredictedClass != -1) {

                            //Also, it is necessary to check if the randomly chosen class is equal
                            //to the real class. In this case, the prediction is correct
                            if (datasetTest.get(numInst).get(i) == 1 && matrixOutputs[numInst][i] == 1) {
                                confusionMatrices[i - indexFirstClass][i - indexFirstClass]++;
                                correctPredictionsLevel++;
                            } else {
                                confusionMatrices[i - indexFirstClass][posPredictedClass - indexFirstClass]++;
                                wrongPredictionsLevel++;
                            }
                        }
                    }//END if (datasetTest.get(numInst).get(i) 
                }//END for (int numInst = 0; 
            }//END for (int i = indexFirstClassLast

            //Calculate accuracy for the current level
            //evalResults[0][ind] = (double) correctPredictionsLevel / (double) (correctPredictionsLevel + wrongPredictionsLevel);

            //Calculate weighted true positive rate for the current level. Also called Recall
            evalResults[0][ind] = calculateWeightedRecallLevel(confusionMatrices, indexFirstClass, indexFirstClassCurrent, indexLastClass);

            //Calculate weighted precision for current level
            evalResults[1][ind] = calculateWeightedPrecisionLevel(confusionMatrices, indexFirstClass, indexFirstClassCurrent, indexLastClass);

            //Calculate weighted f-measure for the current level
            evalResults[2][ind] = calculateWeightedFmeasureLevel(confusionMatrices, indexFirstClass, indexFirstClassCurrent, indexLastClass);

            //Calculate unweighted true positive rate for the current level. Also called Recalll
            evalResults[3][ind] = calculateUnweightedRecallLevel(confusionMatrices, indexFirstClass, indexFirstClassCurrent, indexLastClass);

            //Calculate unweighted precision for the current level.
            evalResults[4][ind] = calculateUnweightedPrecisionLevel(confusionMatrices, indexFirstClass, indexFirstClassCurrent, indexLastClass);

            //Calculate unweighted f-measure for the current level
            evalResults[5][ind] = calculateUnweightedFmeasureLevel(confusionMatrices, indexFirstClass, indexFirstClassCurrent, indexLastClass);

        }//END for (int ind = (numLevels - 1)

        //Calculate accuracy for the leaf classes
        //evalResults[4][0] = calculateAccuracyLeaves(indexesLevels, namesAttributes, confusionMatrices);

        //Calculate weighted true positive rate for leaf classes. Also called Recalll
        evalResults[6][0] = calculateWeightedRecallLeaves(indexesLevels, namesAttributes, confusionMatrices);

        //Calculate weighted precision for leaf classes
        evalResults[7][0] = calculateWeightedPrecisionLeaves(indexesLevels, namesAttributes, confusionMatrices);

        //Calculate weighted f-measure for leaf classes
        evalResults[8][0] = calculateWeightedFmeasureLeaves(indexesLevels, namesAttributes, confusionMatrices);

        //Calculate unweighted recall for leaf classes
        evalResults[9][0] = calculateUnweightedRecallLeaves(indexesLevels, namesAttributes, confusionMatrices);

        //Calculate unweighted precision for leaf classes
        evalResults[10][0] = calculateUnweightedPrecisionLeaves(indexesLevels, namesAttributes, confusionMatrices);

        //Calculate unweighted f-measure for leaf classes
        evalResults[11][0] = calculateUnweightedFmeasureLeaves(indexesLevels, namesAttributes, confusionMatrices);

        saveConfusionMatrices(nameDatasetTest, confusionMatrices, errChoice, numRun, learningAlgorithm, numberEpochs);

        System.out.println("=============================================================");
        for (int i = 0; i < numClasses; i++) {
            for (int j = 0; j < numClasses; j++) {
                System.out.print(confusionMatrices[i][j] + "  ");
            }
            System.out.println();
        }
        System.out.println("=============================================================");

        return evalResults;
    }

    /*===========================================================================
     * Save the results
     *===========================================================================*/
    static void saveResults(String nameDatasetTest,
            int numberEpochs,
            int[][] matrixOutputs,
            double[][] matrixOutputsD,
            double[][] evalResults,
            ArrayList<ArrayList<Integer>> indexesLevels,
            ArrayList<ArrayList<Double>> meanSquareErrors,
            int numLevels,
            int actEpoch,
            int errChoice,
            int numRun,
            int printPredictions,
            ArrayList<String> namesAttributes,
            int learningAlgorithm) {

        Paths paths = new Paths();

        String dirErr = paths.setErrorChoice(errChoice);
        String dirAlg = paths.setLearningAlgorithm(learningAlgorithm);

        paths.setPathsSingleLabel(dirAlg, dirErr, numberEpochs, nameDatasetTest, numRun);

        //String filePath = paths.getFilePath();
        //String filePathD = "Results/" + dirAlg + "/" + dirErr + "/Run" + numRun + "/" + nameDatasetTest + "/" + numberEpochs + "/";
        //String filePredictions = filePath + "/" + "predictions.txt";
        //String filePredictionsD = filePathD + "/" + "predictionsD.txt";
        String fileResults = paths.getFileResults();
        String fileMeanSquareErrorsTraining = paths.getFileMeanSquareErrorsTraining();
        String filePredictions = paths.getFilePredictions();

        //int indexFirstClass = indexesLevels.get(0).get(0);

        try {

            if (printPredictions == 1) {

                //Predictions
                File predictions = new File(filePredictions);
                FileWriter fstreamP = new FileWriter(predictions);
                BufferedWriter outP = new BufferedWriter(fstreamP);

                for (int i = 0; i < matrixOutputs.length; i++) {
                    String predClasses = FunctionsCommon.retrievePrediction(matrixOutputs[i], namesAttributes, indexesLevels);
                    outP.write(predClasses + "\n");
                }
                outP.close();
            }

            //Prediction D
       /*     File predictionsD = new File(filePredictionsD);
             FileWriter fstreamPD = new FileWriter(predictionsD);
             BufferedWriter outPD = new BufferedWriter(fstreamPD);
            
             for(int i=0; i<matrixOutputs.length; i++){
             for(int j=indexFirstClass; j<matrixOutputs[i].length; j++){
             //outP.write(matrixOutputs[i][j] + " ");
             outPD.write(matrixOutputsD[i][j] + " ");
             }
             //outP.write("\n");
             outPD.write("\n");
             }
             //outP.close();
             outPD.close();
             */
            //Results
            File results = new File(fileResults);
            FileWriter fstreamR = new FileWriter(results);
            BufferedWriter outR = new BufferedWriter(fstreamR);

            for (int ind = 0; ind < numLevels; ind++) {
                int level = ind + 1;
                outR.write("Weighted recall level " + level + " = " + evalResults[0][ind] + '\n');
                outR.write("Weighted precision level " + level + " = " + evalResults[1][ind] + '\n');
                outR.write("Weighted f-measure level " + level + " = " + evalResults[2][ind] + '\n');
                outR.write("Unweighted recall level " + level + " = " + evalResults[3][ind] + '\n');
                outR.write("Unweighted precision level " + level + " = " + evalResults[4][ind] + '\n');
                outR.write("Unweighted f-measure level " + level + " = " + evalResults[5][ind] + '\n');
                outR.write("------------------------------------------------------------------------\n");
            }
            outR.write("========================================================================\n");
            outR.write("Weighted recall leaf nodes = " + evalResults[6][0] + '\n');
            outR.write("Weighted precision leaf nodes = " + evalResults[7][0] + '\n');
            outR.write("Weighted f-measure leaf nodes = " + evalResults[8][0] + '\n');
            outR.write("Unweighted recall leaf nodes = " + evalResults[9][0] + '\n');
            outR.write("Unweighted precision leaf nodes = " + evalResults[10][0] + '\n');
            outR.write("Unweighted f-measure leaf nodes = " + evalResults[11][0] + '\n');

            outR.close();

            //Mean square errors training
            File meanSquareErrorsTraining = new File(fileMeanSquareErrorsTraining);
            FileWriter fstreamMSE = new FileWriter(meanSquareErrorsTraining);
            BufferedWriter outMSE = new BufferedWriter(fstreamMSE);

            for (int i = 0; i < meanSquareErrors.size(); i++) {
                int level = i + 1;
                int epoch = actEpoch;
                outMSE.write("=======================================================\n");
                outMSE.write("Training mean square errors at level " + level + "\n");
                outMSE.write("=======================================================\n");

                for (int j = 0; j < meanSquareErrors.get(i).size(); j++) {
                    epoch++;
                    outMSE.write("Epoch = " + epoch + " ---> Error = " + meanSquareErrors.get(i).get(j) + "\n");
                }
            }

            outMSE.close();

        } catch (IOException ioe) {
            ioe.printStackTrace();
        }
    }
}

