/*
 * Functions used in the HMC-LMLP program
 * Fuctions used by multi-label version
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
public class FunctionsMultiLabel implements Serializable {

    /*===========================================================================
     * Get the outputs of the network applying threshold values
     *===========================================================================*/
    static int getOutputThreshold(double outputActFunc, double threshold,
            double thresholdReductionFactor, int actLevel) {

        int output = 0;

        //Threshold reduction factor
        threshold = threshold * Math.pow(thresholdReductionFactor, actLevel);

        if (outputActFunc >= threshold) {
            output = 1;
        } else {
            output = 0;
        }

        return output;
    }

    /*============================================================================
     * Get the threshold values
     *============================================================================*/
    static ArrayList<Double> getThresholdValues(String pathConfigFile) {

        ArrayList<Double> thresholdValues = new ArrayList<Double>();

        try {
            File configFile = new File(pathConfigFile);
            FileReader reader = new FileReader(configFile);
            BufferedReader buffReader = new BufferedReader(reader);

            String regExp = "Threshold values = ";
            Pattern pattern = Pattern.compile(regExp);

            String line = null;
            while ((line = buffReader.readLine()) != null) {
                Matcher m = pattern.matcher(line);
                if (m.find()) {
                    String[] vectorLine1 = line.split("\\[");
                    String[] vectorLine2 = vectorLine1[1].split("\\]");
                    String[] vectorLine3 = vectorLine2[0].split(",");

                    for (int i = 0; i < vectorLine3.length; i++) {
                        thresholdValues.add(Double.parseDouble(vectorLine3[i]));
                    }
                    break;
                }
            }

            buffReader.close();
            reader.close();

        } catch (IOException ioe) {
            ioe.printStackTrace();
        }

        return thresholdValues;
    }

    /*============================================================================
     * Get the threshold reduction parameters
     *============================================================================*/
    static double getThresholdReduction(String pathConfigFile) {

        double thresholdReduction = 0.0;

        try {
            File configFile = new File(pathConfigFile);
            FileReader reader = new FileReader(configFile);
            BufferedReader buffReader = new BufferedReader(reader);

            String regExp = "Threshold reduction = ";
            Pattern pattern = Pattern.compile(regExp);

            String line = null;
            while ((line = buffReader.readLine()) != null) {
                Matcher m = pattern.matcher(line);
                if (m.find()) {
                    String[] vectorLine = line.split(" = ");
                    thresholdReduction = Double.parseDouble(vectorLine[1]);
                    break;
                }
            }

            buffReader.close();
            reader.close();

        } catch (IOException ioe) {
            ioe.printStackTrace();
        }

        return thresholdReduction;
    }

    /*===========================================================================
     * Create directories for output results
     *===========================================================================*/
    static void createDirectories(String nameDatasetTest,
            ArrayList<Integer> numEpochs, ArrayList<Double> thresholdValues,
            int errChoice,
            int numRuns,
            int learningAlgorithm) {

        Paths paths = new Paths();

        String dirErr = paths.setErrorChoice(errChoice);
        String dirAlg = paths.setLearningAlgorithm(learningAlgorithm);

        for (int nRun = 1; nRun <= numRuns; nRun++) {

            paths.setPathsMultiLabel(dirAlg, dirErr, nameDatasetTest, nRun);

            //String strDirectoryDatasetValid = paths.getStrDirectoryDatasetValid();
            //String strDirectoryPRCurvesValid = paths.getStrDirectoryPRCurvesValid();

            String strDirectoryDatasetTest = paths.getStrDirectoryDatasetTest();
            //String strDirectoryPRCurvesTest = paths.getStrDirectoryPRCurvesTest();

            //new File(strDirectoryDatasetValid).mkdirs();
            //new File(strDirectoryPRCurvesValid).mkdirs();

            new File(strDirectoryDatasetTest).mkdirs();
            //new File(strDirectoryPRCurvesTest).mkdirs();

            for (int j = 0; j < thresholdValues.size(); j++) {
                String strDirectoryThresholds = thresholdValues.get(j).toString();
                new File(strDirectoryDatasetTest + "/" + strDirectoryThresholds).mkdir();
            }

            /*
            for(int i=0; i<numEpochs.size(); i++){
            String strDirectoryEpochs = numEpochs.get(i).toString();
            new File(strDirectoryDatasetTest + "/" + strDirectoryEpochs).mkdir();
            new File(strDirectoryDatasetValid + "/" + strDirectoryEpochs).mkdir();
            
            for(int j=0; j<thresholdValues.size(); j++){
            String strDirectoryThresholds = thresholdValues.get(j).toString();
            new File(strDirectoryDatasetTest + "/" + strDirectoryEpochs + "/" + strDirectoryThresholds).mkdir();
            new File(strDirectoryDatasetValid + "/" + strDirectoryEpochs + "/" + strDirectoryThresholds).mkdir();
            }
            }*/
        }
    }

    /*===========================================================================
     * Sigmoidal logistic function
     *===========================================================================*/
    static double sigmoidalLogistic(double inputValue, double a, double threshold) {

        double output = 1 / (1 + Math.exp(-a * inputValue));

        if (output >= threshold) {
            output = 1.0;
        } else {
            output = 0.0;
        }

        return output;
    }

    /*===========================================================================
     * Calculate the output resulted from the first hidden layer and the first level
     *===========================================================================*/
    //Logistic function
    static void outputsFirstHLayerFirstLevel(ArrayList<Integer[]> arrayArchitecture,
            ArrayList<ArrayList<Double[][]>> neuralNet,
            ArrayList<ArrayList<Double[]>> outputs,
            int numInInst, double[][] matrixOutputsD,
            ArrayList<ArrayList<Integer>> indexesLevels, double a,
            ArrayList<ArrayList<ArrayList<Double>>> biasWeights) {

        for (int j = 0; j < (arrayArchitecture.get(0)[1] - arrayArchitecture.get(0)[0]); j++) { //Aqui utiliza exemplos+classes
        //for (int j = 0; j < arrayArchitecture.get(0)[1]; j++) { //Aqui utiliza so exemplos
            for (int i = 0; i < outputs.get(0).get(0).length; i++) {
                outputs.get(1).get(0)[j] += outputs.get(0).get(0)[i] * neuralNet.get(1).get(0)[i][j];
            }
            outputs.get(1).get(0)[j] += 1 * biasWeights.get(1).get(0).get(j);
            matrixOutputsD[numInInst][indexesLevels.get(0).get(j)] = FunctionsCommon.sigmoidalLogistic(outputs.get(1).get(0)[j], a);
            outputs.get(1).get(0)[j] = FunctionsCommon.sigmoidalLogistic(outputs.get(1).get(0)[j], a);
        }
    }

    //Hiperbolic tangent function
    static void outputsFirstHLayerFirstLevel(ArrayList<Integer[]> arrayArchitecture,
            ArrayList<ArrayList<Double[][]>> neuralNet,
            ArrayList<ArrayList<Double[]>> outputs,
            int numInInst, double[][] matrixOutputsD,
            ArrayList<ArrayList<Integer>> indexesLevels, double b, double c,
            ArrayList<ArrayList<ArrayList<Double>>> biasWeights) {

        for (int j = 0; j < (arrayArchitecture.get(0)[1] - arrayArchitecture.get(0)[0]); j++) {
            for (int i = 0; i < outputs.get(0).get(0).length; i++) {
                outputs.get(1).get(0)[j] += outputs.get(0).get(0)[i] * neuralNet.get(1).get(0)[i][j];
            }
            outputs.get(1).get(0)[j] += 1 * biasWeights.get(1).get(0).get(j);
            matrixOutputsD[numInInst][indexesLevels.get(0).get(j)] = FunctionsCommon.sigmoidalHiperbolicTangent(outputs.get(1).get(0)[j], b, c);
            outputs.get(1).get(0)[j] = FunctionsCommon.sigmoidalHiperbolicTangent(outputs.get(1).get(0)[j], b, c);
        }
    }

    /*===========================================================================
     * Error function proposed in Zhang and Zhou, 2006
     *===========================================================================*/
    static double errorZhangZhou(ArrayList<Double> trainingInstance, Double[] outputs,
            ArrayList<Integer> indexesLevels, int indexClass, int k) {

        double error = 0;
        double cardinalityBelong = 0;
        double cardinalityNotBelong = 0;

        //Get cardinality of the sets of classes wich belong and do not belong to the instance
        for (int i = indexesLevels.get(0); i <= indexesLevels.get(indexesLevels.size() - 1); i++) {
            if (trainingInstance.get(i) == 1) {
                cardinalityBelong++;
            } else {
                cardinalityNotBelong++;
            }
        }

        //Verify if the output class belongs to the real set of classes of the instance
        double sum = 0;
        if (trainingInstance.get(indexClass) == 1) {
            for (int i = 0; i < outputs.length; i++) {
                int ind = indexesLevels.get(i);
                if (trainingInstance.get(ind) == 0) {
                    //if(trainingInstance.get(ind) == -1){
                    sum += Math.exp(-(outputs[k] - outputs[i]));
                }
            }
            error = sum / (cardinalityBelong * cardinalityNotBelong);
        } else {
            for (int i = 0; i < outputs.length; i++) {
                int ind = indexesLevels.get(i);
                if (trainingInstance.get(ind) == 1) {
                    sum += Math.exp(-(outputs[i] - outputs[k]));
                }
            }
            error = -sum / (cardinalityBelong * cardinalityNotBelong);
        }

        return error;
    }

    /*===========================================================================
     * Calculate the output resulted from hidden layer and one level
     *===========================================================================*/
    //Logistic function
    static void outputsOneHiddenLayerOneLevel(ArrayList<Integer[]> arrayArchitecture,
            ArrayList<ArrayList<Double[]>> outputs,
            ArrayList<ArrayList<Double[][]>> neuralNet,
            double[][] matrixOutputsD, ArrayList<ArrayList<Integer>> indexesLevels,
            int actLevel, double a, int numInInst,
            ArrayList<ArrayList<ArrayList<Double>>> biasWeights) {

        for (int j = 0; j < (arrayArchitecture.get(0)[actLevel + 1] - arrayArchitecture.get(0)[0]); j++) {//Aqui utiliza exemplos+classes
        //for (int j = 0; j < arrayArchitecture.get(0)[actLevel + 1]; j++) {//Aqui utiliza so exemplos
            for (int i = 0; i < outputs.get(0).get(actLevel).length; i++) {
                outputs.get(1).get(actLevel)[j] += outputs.get(0).get(actLevel)[i] * neuralNet.get(1).get(actLevel)[i][j];
            }
            outputs.get(1).get(actLevel)[j] += 1 * biasWeights.get(1).get(actLevel).get(j);
            matrixOutputsD[numInInst][indexesLevels.get(actLevel).get(j)] = FunctionsCommon.sigmoidalLogistic(outputs.get(1).get(actLevel)[j], a);
            outputs.get(1).get(actLevel)[j] = FunctionsCommon.sigmoidalLogistic(outputs.get(1).get(actLevel)[j], a);
        }
    }
    
    //Hiperbolic tangent function
    static void outputsOneHiddenLayerOneLevel(ArrayList<Integer[]> arrayArchitecture,
            ArrayList<ArrayList<Double[]>> outputs,
            ArrayList<ArrayList<Double[][]>> neuralNet,
            double[][] matrixOutputsD, ArrayList<ArrayList<Integer>> indexesLevels,
            int actLevel, double b, double c, int numInInst,
            ArrayList<ArrayList<ArrayList<Double>>> biasWeights) {

        for (int j = 0; j < (arrayArchitecture.get(0)[actLevel + 1] - arrayArchitecture.get(0)[0]); j++) {
            for (int i = 0; i < outputs.get(0).get(actLevel).length; i++) {
                outputs.get(1).get(actLevel)[j] += outputs.get(0).get(actLevel)[i] * neuralNet.get(1).get(actLevel)[i][j];
            }
            outputs.get(1).get(actLevel)[j] += 1 * biasWeights.get(1).get(actLevel).get(j);
            matrixOutputsD[numInInst][indexesLevels.get(actLevel).get(j)] = FunctionsCommon.sigmoidalHiperbolicTangent(outputs.get(1).get(actLevel)[j], b, c);
            outputs.get(1).get(actLevel)[j] = FunctionsCommon.sigmoidalHiperbolicTangent(outputs.get(1).get(actLevel)[j], b, c);
        }
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

        paths.setPathsMultiLabel(dirAlg, dirErr, numberEpochs, nameDatasetTest, numRun, nameOutput);

        String fileOutput = paths.getFileOutput();

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
     * Obtaing default classes
     *===========================================================================*/
    static double[] obtainDefaultClasses(ArrayList<ArrayList<Double>> datasetTrain,
            ArrayList<ArrayList<Integer>> indexesLevels,
            int numLevels) {

        //Index of classes
        int indexFirstClass = indexesLevels.get(0).get(0);
        int numClassesLast = indexesLevels.get(numLevels - 1).size();
        int indexLastClass = indexesLevels.get(numLevels - 1).get(numClassesLast - 1);

        //Store the counts for each class
        double[] numOccurrences = new double[datasetTrain.get(0).size()];

        for (int numInst = 0; numInst < datasetTrain.size(); numInst++) {

            for (int i = indexFirstClass; i <= indexLastClass; i++) {

                if (datasetTrain.get(numInst).get(i) == 1) {

                    numOccurrences[i]++;
                }
            }
        }

        //Calculate the proportion of examples for each class
        double[] proportions = new double[datasetTrain.get(0).size()];

        for (int i = indexFirstClass; i <= indexLastClass; i++) {
            proportions[i] = numOccurrences[i] / datasetTrain.size();
        }

        return proportions;
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

    static void applyThresholds(int[][] matrixOutputs, double[][] matrixOutputsD,
            ArrayList<ArrayList<Double>> datasetTest,
            ArrayList<ArrayList<Integer>> indexesLevels,
            double threshold,
            double thresholdReductionFactor,
            int numLevels,
            double[] defaultClasses,
            double bestThreshold) {

        //Iterates of all test instances
        for (int numInst = 0; numInst < datasetTest.size(); numInst++) {

            for (int ind = 0; ind < numLevels; ind++) {

                int classified = 0;

                //Index of classes
                int indexFirstClass = indexesLevels.get(ind).get(0);
                int numClassesLast = indexesLevels.get(ind).size();
                int indexLastClass = indexesLevels.get(ind).get(numClassesLast - 1);

                for (int i = indexFirstClass; i <= indexLastClass; i++) {
                    matrixOutputs[numInst][i] = getOutputThreshold(matrixOutputsD[numInst][i],
                            threshold, thresholdReductionFactor, ind);
                    if (matrixOutputs[numInst][i] == 1) {
                        classified = 1;
                    }
                }

                //Verify if no class was predicted for the first level
                //If no, get higher output of the neural network for the first level
                if (classified == 0 && ind == 0) {
                    //if (classified == 0) {
                    for (int i = indexFirstClass; i <= indexLastClass; i++) {
                        matrixOutputs = getHigherOutput(matrixOutputs, matrixOutputsD, numInst, ind,
                                indexFirstClass, indexLastClass);
                    }
                }
            }

            /*
            //Verify if no class was predicted for the first level
            //If no, get higer output of the neural network for the first level
            if (classified == 0) {
            
            for (int ind = 0; ind < numLevels; ind++) {
            
            //Index of classes
            int indexFirstClass = indexesLevels.get(ind).get(0);
            int numClassesLast = indexesLevels.get(ind).size();
            int indexLastClass = indexesLevels.get(ind).get(numClassesLast - 1);
            
            for (int i = indexFirstClass; i <= indexLastClass; i++) {
            matrixOutputs[numInst][i] = getOutputThreshold(matrixOutputsD[numInst][i],
            bestThreshold, thresholdReductionFactor, ind);
            }
            
            //matrixOutputs = getHigherOutput(matrixOutputs, matrixOutputsD, numInst, ind,
            //        indexFirstClass, indexLastClass);
            //matrixOutputs = applyDefaultClasses(matrixOutputs, defaultClasses, numInst, indexesLevels,
            //        numLevels, 0.3, thresholdReductionFactor, ind, indexFirstClass, indexLastClass);
            
            }
            }*/
        }
    }

    /*===========================================================================
     * Get higher output of the neural network for a given level
     *===========================================================================*/
    static int[][] getHigherOutput(int[][] matrixOutputs, double[][] matrixOutputsD,
            int numInst, int ind, int indexFirstClass, int indexLastClass) {


        int higher = FunctionsCommon.getHigherValue(matrixOutputsD, indexFirstClass, indexLastClass, numInst);

        matrixOutputs[numInst][higher] = 1;

        return matrixOutputs;

    }

    /*===========================================================================
     * Apply default classes
     *===========================================================================*/
    static int[][] applyDefaultClasses(int[][] matrixOutputs, double[] defaultClasses, int numInst,
            ArrayList<ArrayList<Integer>> indexesLevels, int numLevels, double threshold,
            double thresholdReductionFactor, int level, int indexFirstClass, int indexLastClass) {


        for (int i = indexFirstClass; i <= indexLastClass; i++) {

            matrixOutputs[numInst][i] = getOutputThreshold(defaultClasses[i],
                    threshold, thresholdReductionFactor, level);

        }


        return matrixOutputs;
    }

    /*===========================================================================
     * Hierarchical Precision and Recall evaluation metrics
     *===========================================================================*/
    static double[][] evaluationPrecRec(ArrayList<ArrayList<Double>> datasetTest,
            int[][] matrixOutputs, ArrayList<ArrayList<Integer>> indexesLevels,
            int numLevels) {

        //Store the results
        double[][] evalResults = new double[5][numLevels];

        //Index of classes
        int indexFirstClass = indexesLevels.get(0).get(0);

        for (int ind = 0; ind < numLevels; ind++) {

            //Sum of predicted and real classes
            double sumIntersection = 0;
            double sumPredicted = 0;
            double sumReal = 0;
            double FP = 0;

            int numClassesLast = indexesLevels.get(ind).size();
            int indexLastClass = indexesLevels.get(ind).get(numClassesLast - 1);

            //Iterates of all test instances
            for (int numInst = 0; numInst < datasetTest.size(); numInst++) {

                for (int i = indexFirstClass; i <= indexLastClass; i++) {

                    if (matrixOutputs[numInst][i] == 1 && datasetTest.get(numInst).get(i) == 1) {
                        sumIntersection++;
                    }
                    if (matrixOutputs[numInst][i] == 1) {
                        sumPredicted++;
                    }

                    if (matrixOutputs[numInst][i] == 1 && datasetTest.get(numInst).get(i) == 0) {
                        FP++;
                    }

                    if (datasetTest.get(numInst).get(i) == 1) {
                        sumReal++;
                    }
                }
            }

            //Hierarchical Precision
            double hPrecision = 0.0;
            if (sumPredicted != 0) {
                hPrecision = sumIntersection / sumPredicted;
            }

            //Hierarchical Recall
            double hRecall = sumIntersection / sumReal;

            evalResults[0][ind] = hPrecision;
            evalResults[1][ind] = hRecall;

            evalResults[2][ind] = sumIntersection; //TP
            evalResults[3][ind] = FP;              //FP
            evalResults[4][ind] = sumReal;         //True

        }

        return evalResults;
    }

    /*===========================================================================
     * Hierarchical Precision and Recall evaluation metrics for each class
     *===========================================================================*/
    static ArrayList<double[]> evaluationPrecRecClasses(ArrayList<ArrayList<Double>> datasetTest,
            int[][] matrixOutputs, ArrayList<ArrayList<Integer>> indexesLevels,
            int numLevels) {

        //Store the results
        ArrayList<double[]> evalResultsClasses = new ArrayList<double[]>();

        //Index of classes
        int indexFirstClass = indexesLevels.get(0).get(0);
        int numClassesLast = indexesLevels.get(numLevels - 1).size();
        int indexLastClass = indexesLevels.get(numLevels - 1).get(numClassesLast - 1);

        //Iterates over all classes
        for (int i = indexFirstClass; i <= indexLastClass; i++) {

            double[] evalResults = new double[5];

            //Sum of predicted and real classes
            double sumIntersection = 0;
            double sumPredicted = 0;
            double sumReal = 0;
            double FP = 0;

            //Iterates over all test instances
            for (int numInst = 0; numInst < datasetTest.size(); numInst++) {

                if (matrixOutputs[numInst][i] == 1 && datasetTest.get(numInst).get(i) == 1) {
                    sumIntersection++;
                }
                if (matrixOutputs[numInst][i] == 1) {
                    sumPredicted++;
                }

                if (matrixOutputs[numInst][i] == 1 && datasetTest.get(numInst).get(i) == 0) {
                    FP++;
                }

                if (datasetTest.get(numInst).get(i) == 1) {
                    sumReal++;
                }
            }

            //Hierarchical Precision
            double hPrecision = 0.0;
            if (sumPredicted != 0) {
                hPrecision = sumIntersection / sumPredicted;
            }

            //Hierarchical Recall
            double hRecall = 0.0;
            if (sumReal != 0) {
                hRecall = sumIntersection / sumReal;
            }

            evalResults[0] = hPrecision;
            evalResults[1] = hRecall;

            evalResults[2] = sumIntersection; //TP
            evalResults[3] = FP;              //FP
            evalResults[4] = sumReal;         //True

            evalResultsClasses.add(evalResults);
        }

        return evalResultsClasses;
    }

    /*===========================================================================
     * Get data values to interpolate two PR points
     *===========================================================================*/
    static double[] getDataInterpolation(double[][] valuesPRA,
            double[][] valuesPRB,
            int numLevels) {

        double[] dataInterpolation = new double[10];
        double localSkew;

        //Values for point A
        double precisionA = valuesPRA[0][numLevels - 1];
        double recallA = valuesPRA[1][numLevels - 1];
        double tpA = valuesPRA[2][numLevels - 1];
        double fpA = valuesPRA[3][numLevels - 1];
        double total = valuesPRA[4][numLevels - 1];

        //Values for point B
        double precisionB = valuesPRB[0][numLevels - 1];
        double recallB = valuesPRB[1][numLevels - 1];
        double tpB = valuesPRB[2][numLevels - 1];
        double fpB = valuesPRB[3][numLevels - 1];

        if ((tpB - tpA) == 0) {
            localSkew = 0;
        } else {
            localSkew = (fpB - fpA) / (tpB - tpA);
        }

        dataInterpolation[0] = localSkew;
        dataInterpolation[1] = tpA;
        dataInterpolation[2] = fpA;
        dataInterpolation[3] = tpB;
        dataInterpolation[4] = fpB;
        dataInterpolation[5] = precisionA;
        dataInterpolation[6] = recallA;
        dataInterpolation[7] = precisionB;
        dataInterpolation[8] = recallB;
        dataInterpolation[9] = total;

        return dataInterpolation;

    }

    static double[] getDataInterpolation(double[] valuesPRA,
            double[] valuesPRB) {

        double[] dataInterpolation = new double[10];
        double localSkew;

        //Values for point A
        double precisionA = valuesPRA[0];
        double recallA = valuesPRA[1];
        double tpA = valuesPRA[2];
        double fpA = valuesPRA[3];
        double total = valuesPRA[4];

        //Values for point B
        double precisionB = valuesPRB[0];
        double recallB = valuesPRB[1];
        double tpB = valuesPRB[2];
        double fpB = valuesPRB[3];

        if ((tpB - tpA) == 0) {
            localSkew = 0;
        } else {
            localSkew = (fpB - fpA) / (tpB - tpA);
        }

        dataInterpolation[0] = localSkew;
        dataInterpolation[1] = tpA;
        dataInterpolation[2] = fpA;
        dataInterpolation[3] = tpB;
        dataInterpolation[4] = fpB;
        dataInterpolation[5] = precisionA;
        dataInterpolation[6] = recallA;
        dataInterpolation[7] = precisionB;
        dataInterpolation[8] = recallB;
        dataInterpolation[9] = total;

        return dataInterpolation;

    }

    /*===========================================================================
     * Interpolate points between two P/R values
     *===========================================================================*/
    static ArrayList<ArrayList<Double>> getPoints(double[] dataInterpolation, int count) {

        ArrayList<ArrayList<Double>> points = new ArrayList<ArrayList<Double>>();
        ArrayList<Double> prec = new ArrayList<Double>();
        ArrayList<Double> reca = new ArrayList<Double>();

        double localSkew = dataInterpolation[0];

        double tpA = dataInterpolation[1];
        double fpA = dataInterpolation[2];
        double tpB = dataInterpolation[3];
        double fpB = dataInterpolation[4];

        double precA = dataInterpolation[5];
        double recaA = dataInterpolation[6];
        double precB = dataInterpolation[7];
        double recaB = dataInterpolation[8];

        double total = dataInterpolation[9];

        double param = tpB - tpA;
        double newPrec;
        double newReca;

        if (count == 0 && recaA > 0) {
            prec.add(precA);
            reca.add(0.0);
        }

        if (count == 0) {
            prec.add(precA);
            reca.add(recaA);
        }

        if (count > 0 && (precA != 0 || recaA != 0 || precB != 0 || recaB != 0)) {
            prec.add(precA);
            reca.add(recaA);
        }

        //if (param < 0) {
        //    param = param * (-1);
        //}
        //if (param > 0 && localSkew > 0) {
 /*       for (int i = 1; i < param; i++) {
        //if ((tpA + i + (fpA + (localSkew * i))) == 0 || (tpA == 0 && fpA == 0)) {
        //    newPrec = 0;
        //    newReca = 0;
        //} else {
        newPrec = (tpA + i) / (tpA + i + (fpA + (localSkew * i)));
        newReca = (tpA + i) / total;
        //}
        prec.add(newPrec);
        reca.add(newReca);
        }
         */      //}

        //Copiado do Clus
        for (int tp = (int) tpA + 1; tp < tpB; tp++) {
            double fp = fpA + localSkew * (tp - tpA);
            newPrec = tp / (tp + fp);
            newReca = tp / total;
            prec.add(newPrec);
            reca.add(newReca);
        }

        prec.add(precB);
        reca.add(recaB);

        points.add(prec);
        points.add(reca);

        return points;

    }

    /*===========================================================================
     * Calculate the area under a curve
     *===========================================================================*/
    static double calculateAreaUnderCurve(ArrayList<ArrayList<Double>> recall,
            ArrayList<ArrayList<Double>> precision) {

        double AUPRC = 0;
        ArrayList<Double> x = new ArrayList<Double>();
        ArrayList<Double> y = new ArrayList<Double>();

        for (int i = 0; i < recall.size(); i++) {
            for (int j = 0; j < recall.get(i).size(); j++) {
                x.add(recall.get(i).get(j));
                y.add(precision.get(i).get(j));
            }
        }

        for (int i = 0; i < x.size() - 1; i++) {
            AUPRC += (x.get(i + 1) - x.get(i)) * y.get(i + 1)
                    + (x.get(i + 1) - x.get(i)) * (y.get(i) - y.get(i + 1)) / 2;
        }

        return AUPRC;

    }

    /*===========================================================================
     * Calculate the AU(PRC)
     *===========================================================================*/
    static double calculateAUPRC(ArrayList<double[][]> valuesPrecisionRecall,
            int numLevels, int errChoice, int learningAlgorithm, String nameDatasetTest, int numRun) {

        double AUPRC = 0;
        ArrayList<ArrayList<Double>> precision = new ArrayList<ArrayList<Double>>();
        ArrayList<ArrayList<Double>> recall = new ArrayList<ArrayList<Double>>();

        int count = 0;

        for (int i = valuesPrecisionRecall.size() - 1; i > 0; i--) {

            //Recover data for interpolation
            double[] dataInterpolation = getDataInterpolation(valuesPrecisionRecall.get(i),
                    valuesPrecisionRecall.get(i - 1),
                    numLevels);

            //Get points between A and B to interpolate
            ArrayList<ArrayList<Double>> points = getPoints(dataInterpolation, count);

            if (i < (valuesPrecisionRecall.size() - 1)) {
                precision.get(precision.size() - 1).remove(precision.get(precision.size() - 1).size() - 1);
                recall.get(recall.size() - 1).remove(recall.get(recall.size() - 1).size() - 1);
            }

            precision.add(points.get(0));
            recall.add(points.get(1));

            count++;

        }

        saveInterpolation(recall, precision, errChoice, learningAlgorithm, nameDatasetTest, numRun);

        AUPRC = calculateAreaUnderCurve(recall, precision);

        return AUPRC;

    }

    static double calculateAUPRC(ArrayList<double[][]> valuesPrecisionRecall, int numLevels) {

        double AUPRC = 0;
        ArrayList<ArrayList<Double>> precision = new ArrayList<ArrayList<Double>>();
        ArrayList<ArrayList<Double>> recall = new ArrayList<ArrayList<Double>>();

        int count = 0;

        for (int i = valuesPrecisionRecall.size() - 1; i > 0; i--) {

            //Recover data for interpolation
            double[] dataInterpolation = getDataInterpolation(valuesPrecisionRecall.get(i),
                    valuesPrecisionRecall.get(i - 1),
                    numLevels);

            //Get points between A and B to interpolate
            ArrayList<ArrayList<Double>> points = getPoints(dataInterpolation, count);

            if (i < (valuesPrecisionRecall.size() - 1)) {
                precision.get(precision.size() - 1).remove(precision.get(precision.size() - 1).size() - 1);
                recall.get(recall.size() - 1).remove(recall.get(recall.size() - 1).size() - 1);
            }

            precision.add(points.get(0));
            recall.add(points.get(1));

            count++;

        }

        AUPRC = calculateAreaUnderCurve(recall, precision);

        return AUPRC;

    }

    /*===========================================================================
     * Calculate the AU(PRC) for classes individually
     *===========================================================================*/
    static ArrayList<double[]> calculateAUPRCClasses(ArrayList<ArrayList<double[]>> valuesPrecisionRecallClasses,
            int numLevels, ArrayList<ArrayList<Integer>> indexesLevels, ArrayList<String> namesAttributes,
            int errChoice, int learningAlgorithm, String nameDatasetTest, int numRun) {

        ArrayList<double[]> AUPRCClasses = new ArrayList<double[]>();
        
        ArrayList<ArrayList<ArrayList<Double>>> interpolatedPrecisionPoints = new ArrayList<ArrayList<ArrayList<Double>>>();
        ArrayList<ArrayList<ArrayList<Double>>> interpolatedRecallPoints = new ArrayList<ArrayList<ArrayList<Double>>>();

        //Index of classes
        int indexFirstClass = indexesLevels.get(0).get(0);
        int numClassesLast = indexesLevels.get(numLevels - 1).size();
        int indexLastClass = indexesLevels.get(numLevels - 1).get(numClassesLast - 1);

        //Iterates over all classes
        int aux = 0;
        for (int ind = indexFirstClass; ind <= indexLastClass; ind++) {

            ArrayList<ArrayList<Double>> precision = new ArrayList<ArrayList<Double>>();
            ArrayList<ArrayList<Double>> recall = new ArrayList<ArrayList<Double>>();
            double[] AUPRC = new double[2];

            int count = 0;
            double totalTrueClass = 0;

            for (int i = valuesPrecisionRecallClasses.size() - 1; i > 0; i--) {

                //Recover data for interpolation
                double[] dataInterpolation = getDataInterpolation(valuesPrecisionRecallClasses.get(i).get(aux),
                        valuesPrecisionRecallClasses.get(i - 1).get(aux));

                totalTrueClass = dataInterpolation[9];

                //Get points between A and B to interpolate
                ArrayList<ArrayList<Double>> points = getPoints(dataInterpolation, count);

                if (i < (valuesPrecisionRecallClasses.size() - 1)) {
                    precision.get(precision.size() - 1).remove(precision.get(precision.size() - 1).size() - 1);
                    recall.get(recall.size() - 1).remove(recall.get(recall.size() - 1).size() - 1);
                }

                precision.add(points.get(0));
                recall.add(points.get(1));

                count++;

            }

            interpolatedPrecisionPoints.add(precision);
            interpolatedRecallPoints.add(recall);

            AUPRC[0] = calculateAreaUnderCurve(recall, precision);
            AUPRC[1] = totalTrueClass;
            AUPRCClasses.add(AUPRC);
            aux++;

        }

        saveInterpolationClasses(interpolatedPrecisionPoints, interpolatedRecallPoints, indexFirstClass, indexLastClass,
                            namesAttributes, errChoice, learningAlgorithm, nameDatasetTest, numRun);
        
        return AUPRCClasses;

    }
    
    /*===========================================================================
     * Save the interpolated points for individual classes (For testing)
     *===========================================================================*/
    static void saveInterpolationClasses(ArrayList<ArrayList<ArrayList<Double>>> interpolatedPrecisionPoints,
            ArrayList<ArrayList<ArrayList<Double>>> interpolatedRecallPoints,
            int indexFirstClass, int indexLastClass, ArrayList<String> namesAttributes,
            int errChoice, int learningAlgorithm, String nameDatasetTest, int numRun) {

                
        Paths paths = new Paths();

        String dirErr = paths.setErrorChoice(errChoice);
        String dirAlg = paths.setLearningAlgorithm(learningAlgorithm);

        paths.setPaths(dirAlg, dirErr, nameDatasetTest, numRun);

        String fileInterpolationsClasses = paths.getFileInterpolationsClasses();

        try {

            File interpolated = new File(fileInterpolationsClasses);
            FileWriter FI = new FileWriter(interpolated);
            BufferedWriter outFI = new BufferedWriter(FI);

            //Iterates over all classes
            int aux = 0;
            for (int ind = indexFirstClass; ind <= indexLastClass; ind++) {

                ArrayList<ArrayList<Double>> precision = interpolatedPrecisionPoints.get(aux);
                ArrayList<ArrayList<Double>> recall = interpolatedRecallPoints.get(aux);

                outFI.write("================== ");
                outFI.write(namesAttributes.get(ind));
                outFI.write(" ==================\n");

                outFI.write("Precision\t\t\t\t\tRecall\n");

                for (int i = 0; i < recall.size(); i++) {
                    for (int j = 0; j < recall.get(i).size(); j++) {

                        outFI.write(precision.get(i).get(j).toString());
                        outFI.write("\t\t\t\t\t" + recall.get(i).get(j).toString() + "\n");
                    }
                }
                aux++;
            }

            outFI.write("================== END ==================\n");

            outFI.close();

        } catch (IOException ioe) {
            ioe.printStackTrace();
        }
    }

    /*===========================================================================
     * Save the interpolated points (For testing)
     *===========================================================================*/
    static void saveInterpolation(ArrayList<ArrayList<Double>> recall,
            ArrayList<ArrayList<Double>> precision, int errChoice,
            int learningAlgorithm, String nameDatasetTest, int numRun) {

        Paths paths = new Paths();

        String dirErr = paths.setErrorChoice(errChoice);
        String dirAlg = paths.setLearningAlgorithm(learningAlgorithm);

        paths.setPaths(dirAlg, dirErr, nameDatasetTest, numRun);

        String fileInterpolations = paths.getFileInterpolations();

        try {

            File interpolated = new File(fileInterpolations);
            FileWriter FI = new FileWriter(interpolated);
            BufferedWriter outFI = new BufferedWriter(FI);

            outFI.write("Precision\t\t\t\t\tRecall\n");

            for (int i = 0; i < recall.size(); i++) {
                for (int j = 0; j < recall.get(i).size(); j++) {

                    outFI.write(precision.get(i).get(j).toString());
                    outFI.write("\t\t\t\t\t" + recall.get(i).get(j).toString() + "\n");
                }
            }

            outFI.close();

        } catch (IOException ioe) {
            ioe.printStackTrace();
        }
    }

    /*===========================================================================
     * Save the predictions (real numbers)
     *===========================================================================*/
    static void savePredictions(String nameDatasetTest,
            double[][] matrixOutputsD,
            ArrayList<ArrayList<Integer>> indexesLevels,
            int numLevels,
            int errChoice,
            int numRun,
            int learningAlgorithm) {

        Paths paths = new Paths();

        String dirErr = paths.setErrorChoice(errChoice);
        String dirAlg = paths.setLearningAlgorithm(learningAlgorithm);

        paths.setPathsMultiLabel(dirAlg, dirErr, nameDatasetTest, numRun);

        String filePredictions = paths.getFilePredictionRealNumbers();

        try {

            int indexFirstClass = indexesLevels.get(0).get(0);

            //Prediction real numbers
            File predictions = new File(filePredictions);
            FileWriter fstreamPD = new FileWriter(predictions);
            BufferedWriter outPD = new BufferedWriter(fstreamPD);

            for (int i = 0; i < matrixOutputsD.length; i++) {
                for (int j = indexFirstClass; j < matrixOutputsD[i].length; j++) {
                    outPD.write(matrixOutputsD[i][j] + " ");
                }
                outPD.write("\n");
            }
            outPD.close();

        } catch (IOException ioe) {
            ioe.printStackTrace();
        }
    }

    /*===========================================================================
     * Save the results
     *===========================================================================*/
    static void saveResults(String nameDatasetTest,
            double threshold,
            int[][] matrixOutputs,
            double[][] matrixOutputsD,
            double[][] evalResults,
            ArrayList<ArrayList<Integer>> indexesLevels,
            int numLevels,
            int errChoice,
            int numRun,
            int printPredictions,
            ArrayList<String> namesAttributes,
            int learningAlgorithm,
            ArrayList<double[]> evalResultsClasses) {

        Paths paths = new Paths();

        String dirErr = paths.setErrorChoice(errChoice);
        String dirAlg = paths.setLearningAlgorithm(learningAlgorithm);

        paths.setPathsMultiLabel(dirAlg, dirErr, nameDatasetTest, numRun, threshold);

        //String filePath = paths.getFilePath();
        //String filePathD = "Results/" + dirAlg + "/" + dirErr + "/Run" + numRun + "/" + nameDatasetTest + "/" + numberEpochs + "/";
        //String filePredictions = filePath + "/" + "predictions.txt";
        //String filePredictionsD = filePathD + "/" + "predictionsD.txt";
        String fileResults = paths.getFileResults();
        //String fileMeanSquareErrorsTraining = paths.getFileMeanSquareErrorsTraining();
        String filePredictions = paths.getFilePredictions();
        String fileResultsClasses = paths.getFileResultsClasses();

        int indexFirstClass = indexesLevels.get(0).get(0);

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
                outR.write("Precision level " + level + " = " + evalResults[0][ind] + '\n');
                outR.write("Recall level " + level + " = " + evalResults[1][ind] + '\n');

                outR.write("\tTrue Positives = " + evalResults[2][ind] + '\n');
                outR.write("\tFalse Positives = " + evalResults[3][ind] + '\n');
                outR.write("\tTotal Real = " + evalResults[4][ind] + '\n');
            }

            outR.close();

            //Results in all classes
            File resultsClasses = new File(fileResultsClasses);
            FileWriter fstreamRClasses = new FileWriter(resultsClasses);
            BufferedWriter outRClasses = new BufferedWriter(fstreamRClasses);

            int aux = 0;
            for (int ind = indexFirstClass; ind < namesAttributes.size(); ind++) {

                //if (evalResultsClasses.get(aux)[2] != 0 || evalResultsClasses.get(aux)[3] != 0) {
                outRClasses.write(namesAttributes.get(ind) + ", ");
                outRClasses.write("Precision = " + evalResultsClasses.get(aux)[0] + ", ");
                outRClasses.write("Recall = " + evalResultsClasses.get(aux)[1] + ", ");
                outRClasses.write("TP = " + evalResultsClasses.get(aux)[2] + ", ");
                outRClasses.write("FP = " + evalResultsClasses.get(aux)[3] + ", ");
                outRClasses.write("Total True = " + evalResultsClasses.get(aux)[4] + "\n");
                //}

                aux++;
            }

            outRClasses.close();

        } catch (IOException ioe) {
            ioe.printStackTrace();
        }
    }

    static void saveResults(double AUPRCTest, double bestAUPRC, int bestEpoch,
            ArrayList<ArrayList<Double[][]>> bestNeuralNet, ArrayList<ArrayList<ArrayList<Double>>> bestBiasWeights,
            double[] trainingTimes, int errChoice, int learningAlgorithm,
            String nameDatasetTest, int numRun,
            ArrayList<ArrayList<Double>> meanSquareErrors,
            int numLevels) {

        Paths paths = new Paths();

        String dirErr = paths.setErrorChoice(errChoice);
        String dirAlg = paths.setLearningAlgorithm(learningAlgorithm);
        paths.setPathsMultiLabel(dirAlg, dirErr, nameDatasetTest, numRun);
        String fileMeanSquareErrorsTraining = paths.getFileMeanSquareErrorsTraining();

        String fileBestResults = paths.getFileBestResults();

        try {

            File results = new File(fileBestResults);
            FileWriter fstreamR = new FileWriter(results);
            BufferedWriter outR = new BufferedWriter(fstreamR);

            outR.write("Best valid AU.PRC = " + bestAUPRC + "\n");
            outR.write("Best testing AU.PRC = " + AUPRCTest + "\n");
            outR.write("Best results obtained at epoch = " + bestEpoch + "\n\n");

            outR.write("--------------- Training time ---------------\n");
            outR.write("Milliseconds = " + trainingTimes[0] + "\n");
            outR.write("Seconds = " + trainingTimes[1] + "\n");
            outR.write("Minutes = " + trainingTimes[2] + "\n");
            outR.write("Hours = " + trainingTimes[3] + "\n");

            outR.close();

            //Save the best Neural Network
            FileOutputStream fileStream = new FileOutputStream(paths.getFileBestNeuralNetwork());
            ObjectOutputStream os = new ObjectOutputStream(fileStream);
            os.writeObject(bestNeuralNet);
            os.close();

            //Save the bias weights of the best neural network
            FileOutputStream fileStreamB = new FileOutputStream(paths.getFileBestBiasWeights());
            ObjectOutputStream osB = new ObjectOutputStream(fileStreamB);
            osB.writeObject(bestBiasWeights);
            osB.close();

            //Mean square errors training
            File meanSquareErrorsTraining = new File(fileMeanSquareErrorsTraining);
            FileWriter fstreamMSE = new FileWriter(meanSquareErrorsTraining);
            BufferedWriter outMSE = new BufferedWriter(fstreamMSE);

            int param = meanSquareErrors.size() / numLevels;
            for (int i = 0; i < numLevels; i++) {
                int level = i + 1;
                int k = i;
                int epoch = 1;
                outMSE.write("=======================================================\n");
                outMSE.write("Training mean square errors at level " + level + "\n");
                outMSE.write("=======================================================\n");
                for (int j = 0; j < param; j++) {

                    for (int l = 0; l < meanSquareErrors.get(k).size(); l++) {
                        outMSE.write("Epoch = " + epoch + " ---> Error = " + meanSquareErrors.get(k).get(l) + "\n");
                        epoch++;
                    }
                    k += numLevels;
                }
            }

            outMSE.close();

        } catch (IOException ioe) {
            ioe.printStackTrace();
        }
    }

    static void saveResults(String nameDatasetTest,
            int learningAlgorithm,
            int errChoice,
            double[] meanSdAUPRCValid,
            double[] meanSdAUPRCTest,
            double[] meanMsTimes,
            double[] meanSTimes,
            double[] meanMTimes,
            double[] meanHTimes,
            double[] meanEpochs) {

        Paths paths = new Paths();
        String dirAlg = paths.setLearningAlgorithm(learningAlgorithm);
        String dirErr = paths.setErrorChoice(errChoice);
        paths.setPathsMultilabel(dirAlg, dirErr);
        String strDirectoryAlgorithm = paths.getStrDirectoryAlgorithm() + "/" + nameDatasetTest + ".means.txt";

        try {

            File results = new File(strDirectoryAlgorithm);
            FileWriter fstreamR = new FileWriter(results);
            BufferedWriter outR = new BufferedWriter(fstreamR);

            outR.write("Mean (Sd) valid AU.PRC = " + meanSdAUPRCValid[0] + " (" + meanSdAUPRCValid[1] + ")\n");
            outR.write("Mean (Sd) test AU.PRC = " + meanSdAUPRCTest[0] + " (" + meanSdAUPRCTest[1] + ")\n");
            outR.write("Mean (Sd) training epochs of best results = " + meanEpochs[0] + " (" + meanEpochs[1] + ")\n\n");

            outR.write("----------------------- Mean training time -----------------------\n");
            outR.write("Mean (Sd) milliseconds = " + meanMsTimes[0] + " (" + meanMsTimes[1] + ")\n");
            outR.write("Mean (Sd) seconds = " + meanSTimes[0] + " (" + meanSTimes[1] + ")\n");
            outR.write("Mean (Sd) minutes = " + meanMTimes[0] + " (" + meanMTimes[1] + ")\n");
            outR.write("Mean (Sd) hours = " + meanHTimes[0] + " (" + meanHTimes[1] + ")\n");

            outR.close();

        } catch (IOException ioe) {
            ioe.printStackTrace();
        }

    }

    /*===========================================================================
     * Save the AUPRC of all classes
     *===========================================================================*/
    static void saveAUPRCClasses(ArrayList<double[]> AUPRCClasses,
            ArrayList<ArrayList<Integer>> indexesLevels,
            String nameDatasetTest,
            ArrayList<String> namesAttributes,
            int numRun, int numLevels,
            int errChoice, int learningAlgorithm) {

        Paths paths = new Paths();

        String dirErr = paths.setErrorChoice(errChoice);
        String dirAlg = paths.setLearningAlgorithm(learningAlgorithm);

        paths.setPathsMultiLabel(dirAlg, dirErr, nameDatasetTest, numRun);

        String fileResultsAUPRCClasses = paths.getFileBestResultsAUPRCClasses();

        //Index of classes
        int indexFirstClass = indexesLevels.get(0).get(0);
        int numClassesLast = indexesLevels.get(numLevels - 1).size();
        int indexLastClass = indexesLevels.get(numLevels - 1).get(numClassesLast - 1);

        try {
            //Results in all classes
            File resultsClasses = new File(fileResultsAUPRCClasses);
            FileWriter fstreamRClasses = new FileWriter(resultsClasses);
            BufferedWriter outRClasses = new BufferedWriter(fstreamRClasses);

            int aux = 0;
            for (int ind = indexFirstClass; ind <= indexLastClass; ind++) {

                //if (evalResultsClasses.get(aux)[2] != 0 || evalResultsClasses.get(aux)[3] != 0) {
                outRClasses.write(namesAttributes.get(ind) + ", ");
                outRClasses.write("AUPRC = " + AUPRCClasses.get(aux)[0] + "\n");
                //}

                aux++;
            }

            outRClasses.close();

        } catch (IOException ioe) {
            ioe.printStackTrace();
        }

    }

    /*===========================================================================
     * Save the means of AUPRC of all classes
     *===========================================================================*/
    static void saveMeansAUPRCClasses(ArrayList<double[]> meansAllAUPRCClasses,
            String nameDatasetTest, ArrayList<ArrayList<Integer>> indexesLevels,
            int numLevels, ArrayList<String> namesAttributes, int errChoice, int learningAlgorithm, int sizeDataset) {

        Paths paths = new Paths();
        String dirAlg = paths.setLearningAlgorithm(learningAlgorithm);
        String dirErr = paths.setErrorChoice(errChoice);
        paths.setPathsMultilabel(dirAlg, dirErr);
        String strDirectoryAlgorithm = paths.getStrDirectoryAlgorithm() + "/" + nameDatasetTest + ".meansClasses.txt";

        //Index of classes
        int indexFirstClass = indexesLevels.get(0).get(0);
        int numClassesLast = indexesLevels.get(numLevels - 1).size();
        int indexLastClass = indexesLevels.get(numLevels - 1).get(numClassesLast - 1);


        try {
            //Results in all classes
            File resultsClasses = new File(strDirectoryAlgorithm);
            FileWriter fstreamRClasses = new FileWriter(resultsClasses);
            BufferedWriter outRClasses = new BufferedWriter(fstreamRClasses);

            int aux = 0;
            for (int ind = indexFirstClass; ind <= indexLastClass; ind++) {

                double freqClass = meansAllAUPRCClasses.get(aux)[2] / sizeDataset;

                //if (evalResultsClasses.get(aux)[2] != 0 || evalResultsClasses.get(aux)[3] != 0) {
                outRClasses.write(namesAttributes.get(ind) + ", ");
                outRClasses.write("Mean AUPRC = " + meansAllAUPRCClasses.get(aux)[0]);
                outRClasses.write(" (" + meansAllAUPRCClasses.get(aux)[1] + ")");
                outRClasses.write(", Freq:" + freqClass + "\n");
                //}

                aux++;
            }

            outRClasses.close();

        } catch (IOException ioe) {
            ioe.printStackTrace();
        }

    }
}
