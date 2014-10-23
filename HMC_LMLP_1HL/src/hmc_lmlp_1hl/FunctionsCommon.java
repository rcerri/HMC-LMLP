/*
 * Functions used in the HMC-LMLP program
 * Common function used by multi-label and single-label version
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
public class FunctionsCommon {

    /*===========================================================================
     * Save the means of AUPRC of all classes
     *===========================================================================*/
    static void saveMeansAUPRCClasses(ArrayList<double[]> meansAllAUPRCClasses,
            String nameDatasetTest, ArrayList<ArrayList<Integer>> indexesLevels,
            int numLevels, ArrayList<String> namesAttributes, int errChoice, int learningAlgorithm, int sizeDataset) {

        Paths paths = new Paths();
        String dirAlg = paths.setLearningAlgorithm(learningAlgorithm);
        String dirErr = paths.setErrorChoice(errChoice);
        paths.setPaths(dirAlg, dirErr);
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

        paths.setPaths(dirAlg, dirErr, nameDatasetTest, numRun);

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

        paths.setPaths(dirAlg, dirErr, nameDatasetTest, numRun, threshold);

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
        paths.setPaths(dirAlg, dirErr, nameDatasetTest, numRun);
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
        paths.setPaths(dirAlg, dirErr);
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
     * Get a percentage of the training dataset as validation
     *===========================================================================*/
    static ArrayList<ArrayList<Double>> getValidDataset(ArrayList<ArrayList<Double>> datasetTrain) {

        ArrayList<ArrayList<Double>> datasetValid = new ArrayList<ArrayList<Double>>();
        ArrayList<Integer> indexes = new ArrayList<Integer>();

        double percentageValid = 0.3;
        int numExamplesValid = (int) (percentageValid * datasetTrain.size());

        Random generator = new Random();

        while (datasetValid.size() < numExamplesValid) {
            int index = generator.nextInt(datasetTrain.size());
            if (indexes.contains(index) == false) {
                datasetValid.add(datasetTrain.get(index));
                indexes.add(index);
            }
        }

        Collections.sort(indexes, Collections.reverseOrder());
        for (int i = 0; i < indexes.size(); i++) {
            datasetTrain.remove((int) indexes.get(i));
        }

        return datasetValid;
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

        paths.setPaths(dirAlg, dirErr, nameDatasetTest, numRun);

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

            paths.setPaths(dirAlg, dirErr, nameDatasetTest, nRun);

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


    /*============================================================================
     * Get the learning algorithm chosen
     *============================================================================*/
    static int getLearningAlgorithm(String pathConfigFile) {

        int learningAlgorithm = 0;

        try {
            File configFile = new File(pathConfigFile);
            FileReader reader = new FileReader(configFile);
            BufferedReader buffReader = new BufferedReader(reader);

            String regExp = "Algorithm = ";
            Pattern pattern = Pattern.compile(regExp);

            String line = null;
            while ((line = buffReader.readLine()) != null) {
                Matcher m = pattern.matcher(line);
                if (m.find()) {
                    String[] vectorLine = line.split(" = ");
                    learningAlgorithm = Integer.parseInt(vectorLine[1]);
                    break;
                }
            }

            buffReader.close();
            reader.close();

        } catch (IOException ioe) {
            ioe.printStackTrace();
        }

        return learningAlgorithm;
    }

    /*============================================================================
     * Get the hierarchy type: DAG or Tree
     *============================================================================*/
    static String getHierarchyType(String pathConfigFile) {

        String hierarchyType = "";

        try {
            File configFile = new File(pathConfigFile);
            FileReader reader = new FileReader(configFile);
            BufferedReader buffReader = new BufferedReader(reader);

            String regExp = "Hierarchy type = ";
            Pattern pattern = Pattern.compile(regExp);

            String line = null;
            while ((line = buffReader.readLine()) != null) {
                Matcher m = pattern.matcher(line);
                if (m.find()) {
                    String[] vectorLine = line.split(" = ");
                    hierarchyType = vectorLine[1];
                    break;
                }
            }

            buffReader.close();
            reader.close();

        } catch (IOException ioe) {
            ioe.printStackTrace();
        }

        return hierarchyType;
    }

    /*============================================================================
     * Get the file which contains the DAG relationships
     *============================================================================*/
    static String getDAGstructure(String pathDatasets, String pathConfigFile) {

        String fileDAGstructure = "";
        String DAGstructure = "";

        try {
            File configFile = new File(pathConfigFile);
            FileReader reader = new FileReader(configFile);
            BufferedReader buffReader = new BufferedReader(reader);

            String regExp = "DAG relationships = ";
            Pattern pattern = Pattern.compile(regExp);

            String line = null;
            while ((line = buffReader.readLine()) != null) {
                Matcher m = pattern.matcher(line);
                if (m.find()) {
                    String[] vectorLine = line.split(" = ");
                    fileDAGstructure = vectorLine[1];
                    break;
                }
            }

            buffReader.close();
            reader.close();

            File DAGFile = new File(pathDatasets + fileDAGstructure);
            reader = new FileReader(DAGFile);
            buffReader = new BufferedReader(reader);

            DAGstructure = buffReader.readLine();

            buffReader.close();
            reader.close();

        } catch (IOException ioe) {
            ioe.printStackTrace();
        }

        return DAGstructure;
    }

    /*============================================================================
     * Get the num of epochs to train. It can be an array of values
     *============================================================================*/
    static ArrayList<Integer> getNumEpochs(String pathConfigFile) {

        ArrayList<Integer> numEpochs = new ArrayList<Integer>();

        try {
            File configFile = new File(pathConfigFile);
            FileReader reader = new FileReader(configFile);
            BufferedReader buffReader = new BufferedReader(reader);

            String regExp = "Training epochs = ";
            Pattern pattern = Pattern.compile(regExp);

            String line = null;
            while ((line = buffReader.readLine()) != null) {
                Matcher m = pattern.matcher(line);
                if (m.find()) {
                    String[] vectorLine1 = line.split("\\[");
                    String[] vectorLine2 = vectorLine1[1].split("\\]");
                    String[] vectorLine3 = vectorLine2[0].split(",");

                    for (int i = 0; i < vectorLine3.length; i++) {
                        numEpochs.add(Integer.parseInt(vectorLine3[i]));
                    }
                    break;
                }
            }

            buffReader.close();
            reader.close();

        } catch (IOException ioe) {
            ioe.printStackTrace();
        }

        return numEpochs;
    }

    /*===========================================================================
     * Dataset is multi-label or single-label?
     *===========================================================================*/
    static int getMultiLabel(String pathConfigFile) {

        int multilabel = 0;

        try {
            File configFile = new File(pathConfigFile);
            FileReader reader = new FileReader(configFile);
            BufferedReader buffReader = new BufferedReader(reader);

            String regExp = "Multi-Label = ";
            Pattern pattern = Pattern.compile(regExp);

            String line = null;
            while ((line = buffReader.readLine()) != null) {
                Matcher m = pattern.matcher(line);
                if (m.find()) {
                    String[] vectorLine = line.split(" = ");
                    multilabel = Integer.parseInt(vectorLine[1]);
                    break;
                }
            }

            buffReader.close();
            reader.close();

        } catch (IOException ioe) {
            ioe.printStackTrace();
        }

        return multilabel;
    }

    /*============================================================================
     * Get the values of learning rate and momentum constant
     *============================================================================*/
    static ArrayList<ArrayList<Double>> getMomentumAndLearning(String pathConfigFile) {

        ArrayList<ArrayList<Double>> momentumAndLearning = new ArrayList<ArrayList<Double>>();

        try {
            File configFile = new File(pathConfigFile);
            FileReader reader = new FileReader(configFile);
            BufferedReader buffReader = new BufferedReader(reader);

            String[] regExpV = {"Learning rate = ", "Momentum constant = "};

            for (int i = 0; i < regExpV.length; i++) {
                String regExp = regExpV[i];
                Pattern pattern = Pattern.compile(regExp);
                ArrayList<Double> data = new ArrayList<Double>();

                String line = null;
                while ((line = buffReader.readLine()) != null) {
                    Matcher m = pattern.matcher(line);
                    if (m.find()) {
                        String[] vectorLine1 = line.split("\\[");
                        String[] vectorLine2 = vectorLine1[1].split("\\]");
                        String[] vectorLine3 = vectorLine2[0].split(",");

                        for (int j = 0; j < vectorLine3.length; j++) {
                            data.add(Double.parseDouble(vectorLine3[j]));
                        }
                        momentumAndLearning.add(data);
                        break;
                    }
                }
            }

            buffReader.close();
            reader.close();

        } catch (IOException ioe) {
            ioe.printStackTrace();
        }

        return momentumAndLearning;
    }

    /*============================================================================
     * Get the error function chosen
     *============================================================================*/
    static double getWeightDecay(String pathConfigFile) {

        double weightDecay = 0.0;

        try {
            File configFile = new File(pathConfigFile);
            FileReader reader = new FileReader(configFile);
            BufferedReader buffReader = new BufferedReader(reader);

            String regExp = "Weight decay = ";
            Pattern pattern = Pattern.compile(regExp);

            String line = null;
            while ((line = buffReader.readLine()) != null) {
                Matcher m = pattern.matcher(line);
                if (m.find()) {
                    String[] vectorLine = line.split(" = ");
                    weightDecay = Double.parseDouble(vectorLine[1]);
                    break;
                }
            }

            buffReader.close();
            reader.close();

        } catch (IOException ioe) {
            ioe.printStackTrace();
        }

        return weightDecay;
    }

    /*===========================================================================
     * Verify weight decay parameter. If equals -1, the weight decay will be
     * 1/(2*learningRate*numMaxEpochs), otherwise it will remain the value chosen
     * by the user.
     *===========================================================================*/
    static double verifyWeightDecay(double weightDecay, double learningRate, int numMaxEpochs) {

        if (weightDecay == -1) {
            weightDecay = 1 / (2 * learningRate * numMaxEpochs);
        }

        return (weightDecay);
    }

    /*============================================================================
     * Get the values of parameters of the resilient back-propagation
     *============================================================================*/
    static ArrayList<Double> getRpropParam(String pathConfigFile) {

        ArrayList<Double> rpropParam = new ArrayList<Double>();

        try {
            File configFile = new File(pathConfigFile);
            FileReader reader = new FileReader(configFile);
            BufferedReader buffReader = new BufferedReader(reader);

            String[] regExpV = {"Increase factor = ", "Decrease factor = ", "Initial delta = ", "Max delta = ", "Min delta = "};

            for (int i = 0; i < regExpV.length; i++) {
                String regExp = regExpV[i];
                Pattern pattern = Pattern.compile(regExp);
                ArrayList<Double> data = new ArrayList<Double>();

                String line = null;
                while ((line = buffReader.readLine()) != null) {
                    Matcher m = pattern.matcher(line);
                    if (m.find()) {
                        String[] vectorLine = line.split(" = ");
                        rpropParam.add(Double.parseDouble(vectorLine[1]));
                        break;
                    }
                }
            }

            buffReader.close();
            reader.close();

        } catch (IOException ioe) {
            ioe.printStackTrace();
        }

        return rpropParam;
    }

    /*============================================================================
     * Get the number of runs chosen
     *============================================================================*/
    static int getNumberRuns(String pathConfigFile) {

        int numberRuns = 0;

        try {
            File configFile = new File(pathConfigFile);
            FileReader reader = new FileReader(configFile);
            BufferedReader buffReader = new BufferedReader(reader);

            String regExp = "Number of runs = ";
            Pattern pattern = Pattern.compile(regExp);

            String line = null;
            while ((line = buffReader.readLine()) != null) {
                Matcher m = pattern.matcher(line);
                if (m.find()) {
                    String[] vectorLine = line.split(" = ");
                    numberRuns = Integer.parseInt(vectorLine[1]);
                    break;
                }
            }

            buffReader.close();
            reader.close();

        } catch (IOException ioe) {
            ioe.printStackTrace();
        }

        return numberRuns;
    }

    /*============================================================================
     * Get the num of levels of the data
     *============================================================================*/
    static int getNumLevels(String pathConfigFile) {

        int numLevels = 0;

        try {
            File configFile = new File(pathConfigFile);
            FileReader reader = new FileReader(configFile);
            BufferedReader buffReader = new BufferedReader(reader);

            String regExp = "Hierarchical levels = ";
            Pattern pattern = Pattern.compile(regExp);

            String line = null;
            while ((line = buffReader.readLine()) != null) {
                Matcher m = pattern.matcher(line);
                if (m.find()) {
                    String[] vectorLine = line.split(" = ");
                    numLevels = Integer.parseInt(vectorLine[1]);
                    break;
                }
            }

            buffReader.close();
            reader.close();

        } catch (IOException ioe) {
            ioe.printStackTrace();
        }

        return numLevels;
    }

    /*============================================================================
     * Get the names of the train, valid and test datasets
     *============================================================================*/
    static ArrayList<ArrayList<String>> getNamesData(String pathConfigFile) {

        ArrayList<ArrayList<String>> namesData = new ArrayList<ArrayList<String>>();

        try {
            File configFile = new File(pathConfigFile);
            FileReader reader = new FileReader(configFile);
            BufferedReader buffReader = new BufferedReader(reader);

            String[] regExpV = {"Dataset train = ", "Dataset valid = ", "Dataset test = "};

            for (int i = 0; i < regExpV.length; i++) {

                ArrayList<String> arrayNamesData = new ArrayList<String>();

                String regExp = regExpV[i];
                Pattern pattern = Pattern.compile(regExp);

                String line = null;
                while ((line = buffReader.readLine()) != null) {
                    Matcher m = pattern.matcher(line);
                    if (m.find()) {
                        String[] vectorLine1 = line.split("\\[");
                        String[] vectorLine2 = vectorLine1[1].split("\\]");
                        String[] vectorLine3 = vectorLine2[0].split(",");

                        for (int j = 0; j < vectorLine3.length; j++) {
                            arrayNamesData.add(vectorLine3[j]);
                        }

                        namesData.add(arrayNamesData);

                        break;
                    }
                }
            }

            buffReader.close();
            reader.close();

        } catch (IOException ioe) {
            ioe.printStackTrace();
        }

        return namesData;
    }

    /*============================================================================
     * Get the path where the datasets are located
     *============================================================================*/
    static String getPathDatasets(String pathConfigFile) {

        String pathDatasets = "";

        try {
            File configFile = new File(pathConfigFile);
            FileReader reader = new FileReader(configFile);
            BufferedReader buffReader = new BufferedReader(reader);

            String regExp = "Path datasets = ";
            Pattern pattern = Pattern.compile(regExp);

            String line = null;
            while ((line = buffReader.readLine()) != null) {
                Matcher m = pattern.matcher(line);
                if (m.find()) {
                    String[] vectorLine = line.split(" = ");
                    pathDatasets = vectorLine[1];
                    break;
                }
            }

            buffReader.close();
            reader.close();

        } catch (IOException ioe) {
            ioe.printStackTrace();
        }

        return pathDatasets;
    }

    /*============================================================================
     * Get the pertentage of hidden units
     *============================================================================*/
    static ArrayList<Double> getPercentageHiddenUnits(String pathConfigFile) {

        ArrayList<Double> percentage = new ArrayList<Double>();

        try {
            File configFile = new File(pathConfigFile);
            FileReader reader = new FileReader(configFile);
            BufferedReader buffReader = new BufferedReader(reader);

            String regExp = "Percentage hidden units = ";
            Pattern pattern = Pattern.compile(regExp);

            String line = null;
            while ((line = buffReader.readLine()) != null) {
                Matcher m = pattern.matcher(line);
                if (m.find()) {
                    String[] vectorLine1 = line.split("\\[");
                    String[] vectorLine2 = vectorLine1[1].split("\\]");
                    String[] vectorLine3 = vectorLine2[0].split(",");

                    for (int i = 0; i < vectorLine3.length; i++) {
                        percentage.add(Double.parseDouble(vectorLine3[i]));
                    }

                    percentage.add(0.0);

                    break;
                }
            }

            buffReader.close();
            reader.close();

        } catch (IOException ioe) {
            ioe.printStackTrace();
        }

        return percentage;
    }

    /*============================================================================
     * Get the error function chosen
     *============================================================================*/
    static int getErrorChoice(String pathConfigFile) {

        int errorChoice = 0;

        try {
            File configFile = new File(pathConfigFile);
            FileReader reader = new FileReader(configFile);
            BufferedReader buffReader = new BufferedReader(reader);

            String regExp = "Error function = ";
            Pattern pattern = Pattern.compile(regExp);

            String line = null;
            while ((line = buffReader.readLine()) != null) {
                Matcher m = pattern.matcher(line);
                if (m.find()) {
                    String[] vectorLine = line.split(" = ");
                    errorChoice = Integer.parseInt(vectorLine[1]);
                    break;
                }
            }

            buffReader.close();
            reader.close();

        } catch (IOException ioe) {
            ioe.printStackTrace();
        }

        return errorChoice;
    }

    /*============================================================================
     * Get the parameter which indicates if predictions should be printed
     *============================================================================*/
    static int getPrintPredictions(String pathConfigFile) {

        int printPredictions = 0;

        try {
            File configFile = new File(pathConfigFile);
            FileReader reader = new FileReader(configFile);
            BufferedReader buffReader = new BufferedReader(reader);

            String regExp = "Print Predictions = ";
            Pattern pattern = Pattern.compile(regExp);

            String line = null;
            while ((line = buffReader.readLine()) != null) {
                Matcher m = pattern.matcher(line);
                if (m.find()) {
                    String[] vectorLine = line.split(" = ");
                    printPredictions = Integer.parseInt(vectorLine[1]);
                    break;
                }
            }

            buffReader.close();
            reader.close();

        } catch (IOException ioe) {
            ioe.printStackTrace();
        }

        return printPredictions;
    }

    /*===========================================================================
     * Get the names of the classes in the dataset
     *===========================================================================*/
    static ArrayList<String> getNamesAttributes(String nameDataset) {

        ArrayList<String> namesAttributes = new ArrayList<String>();
        String pathDataset = nameDataset;

        try {
            File fileDataset = new File(pathDataset);
            FileReader reader = new FileReader(fileDataset);
            BufferedReader buffReader = new BufferedReader(reader);

            String line = null;
            line = buffReader.readLine();
            String[] vectorLine = line.split(",");

            String regExp = "_level";
            Pattern pattern = Pattern.compile(regExp);

            for (int i = 0; i < vectorLine.length; i++) {
                String attribute = vectorLine[i].replace("\"", "");
                Matcher m = pattern.matcher(attribute);
                if (m.find()) {
                    String[] vectorClass = attribute.split("_");
                    namesAttributes.add(vectorClass[1]);
                } else {
                    namesAttributes.add(attribute);
                }
            }

            buffReader.close();
            reader.close();

        } catch (IOException ioe) {
            ioe.printStackTrace();
        }

        return namesAttributes;
    }

    /*===========================================================================
     * Get the indexes of the classes in each hierarchical level
     *===========================================================================*/
    static ArrayList<ArrayList<Integer>> getIndexesLevels(String nameDataset, int numLevels) {

        ArrayList<ArrayList<Integer>> indexesLevels = new ArrayList<ArrayList<Integer>>();
        String pathDataset = nameDataset;

        try {
            File fileDataset = new File(pathDataset);
            FileReader reader = new FileReader(fileDataset);
            BufferedReader buffReader = new BufferedReader(reader);

            String line = null;
            line = buffReader.readLine();
            String[] vectorLine = line.split(",");

            for (int i = 1; i <= numLevels; i++) {
                String regExp = "_level" + i + "$";
                Pattern pattern = Pattern.compile(regExp);
                ArrayList<Integer> indexes = new ArrayList<Integer>();
                for (int j = 0; j < vectorLine.length; j++) {
                    String attribute = vectorLine[j].replace("\"", "");
                    Matcher m = pattern.matcher(attribute);
                    if (m.find()) {
                        indexes.add(j);
                    }
                }
                indexesLevels.add(indexes);
            }

            buffReader.close();
            reader.close();

        } catch (IOException ioe) {
            ioe.printStackTrace();
        }

        return indexesLevels;
    }

    /*============================================================================
     * Read the dataset file
     *============================================================================*/
    static ArrayList<ArrayList<Double>> readDataset(String nameDataset) {

        ArrayList<ArrayList<Double>> dataset = new ArrayList<ArrayList<Double>>();
        String pathDataset = nameDataset;

        try {
            File fileDataset = new File(pathDataset);
            FileReader reader = new FileReader(fileDataset);
            BufferedReader buffReader = new BufferedReader(reader);

            String line = null;
            line = buffReader.readLine();

            while ((line = buffReader.readLine()) != null) {
                String[] vectorLine = line.split(",");
                ArrayList<Double> vectorDouble = new ArrayList<Double>();
                for (int i = 0; i < vectorLine.length; i++) {
                    vectorDouble.add(Double.parseDouble(vectorLine[i]));
                }
                dataset.add(vectorDouble);
            }
            buffReader.close();
            reader.close();

        } catch (IOException ioe) {
            ioe.printStackTrace();
        }

        return dataset;
    }

    /*===========================================================================
     * Initialize the weights of the neural network
     *===========================================================================*/
    static ArrayList<ArrayList<Double[][]>> initializeWeights(ArrayList<Integer[]> arrayArchitecture) {

        Random randomGenerator = new Random();
        ArrayList<ArrayList<Double[][]>> neuralNet = new ArrayList<ArrayList<Double[][]>>();

        ArrayList<Double[][]> neuralNet1 = new ArrayList<Double[][]>();
        ArrayList<Double[][]> neuralNet2 = new ArrayList<Double[][]>();

        for (int i = 0; i < (arrayArchitecture.get(0).length - 1); i++) {
            Double matrixWeights[][];

            //Weights between an input layer and a hidden layer
            int numRows = arrayArchitecture.get(0)[i];//Aqui utiliza exemplos+classes
            //int numRows = arrayArchitecture.get(0)[0];//Aqui utiliza so exemplos
            int numColumns = arrayArchitecture.get(1)[i];
            matrixWeights = new Double[numRows][numColumns];

            for (int j = 0; j < numRows; j++) {
                for (int k = 0; k < numColumns; k++) {
                    //matrixWeights[j][k] = randomGenerator.nextDouble();
                    //matrixWeights[j][k] = 2 * randomGenerator.nextDouble() - 1;
                    matrixWeights[j][k] = 0.2 * randomGenerator.nextDouble() - 0.1;
                    //matrixWeights[j][k] = 0.0;
                }
            }
            neuralNet1.add(matrixWeights);

            //Weights between a hidden layer and an output layer
            numRows = arrayArchitecture.get(1)[i];
            //if(i < (arrayArchitecture.get(0).length-2)){
            numColumns = arrayArchitecture.get(0)[i + 1] - arrayArchitecture.get(0)[0]; //Aqui utiliza exemplos+classes
            //numColumns = arrayArchitecture.get(0)[i + 1]; //Aqui utiliza so exemplos
            //}else{
            //numColumns = arrayArchitecture.get(0)[i+1];
            //}
            matrixWeights = new Double[numRows][numColumns];

            for (int j = 0; j < numRows; j++) {
                for (int k = 0; k < numColumns; k++) {
                    //matrixWeights[j][k] = randomGenerator.nextDouble();
                    //matrixWeights[j][k] = 2 * randomGenerator.nextDouble() - 1;
                    matrixWeights[j][k] = 0.2 * randomGenerator.nextDouble() - 0.1;
                    //matrixWeights[j][k] = 0.0;
                }
            }
            neuralNet2.add(matrixWeights);
        }

        neuralNet.add(neuralNet1);
        neuralNet.add(neuralNet2);

        return neuralNet;
    }

    /*===========================================================================
     * Copy the weights of a neural network
     *===========================================================================*/
    static ArrayList<ArrayList<Double[][]>> copyNeuralNet(ArrayList<ArrayList<Double[][]>> neuralNet,
            ArrayList<ArrayList<Double[][]>> initialNeuralNet,
            ArrayList<Integer[]> arrayArchitecture) {

        for (int i = 0; i < initialNeuralNet.size(); i++) {
            for (int j = 0; j < initialNeuralNet.get(i).size(); j++) {
                for (int k = 0; k < initialNeuralNet.get(i).get(j).length; k++) {
                    for (int l = 0; l < initialNeuralNet.get(i).get(j)[k].length; l++) {
                        neuralNet.get(i).get(j)[k][l] = initialNeuralNet.get(i).get(j)[k][l];
                    }
                }
            }
        }

        return neuralNet;
    }

    /*===========================================================================
     * Copy the bias weights of a neural network
     *===========================================================================*/
    static ArrayList<ArrayList<ArrayList<Double>>> copyBiasWeights(ArrayList<ArrayList<ArrayList<Double>>> initialBiasWeights,
            ArrayList<Integer[]> arrayArchitecture) {

        ArrayList<ArrayList<ArrayList<Double>>> biasWeights = new ArrayList<ArrayList<ArrayList<Double>>>();

        for (int i = 0; i < initialBiasWeights.size(); i++) {
            ArrayList<ArrayList<Double>> weights1 = new ArrayList<ArrayList<Double>>();
            for (int j = 0; j < initialBiasWeights.get(i).size(); j++) {
                ArrayList<Double> weights2 = new ArrayList<Double>();
                for (int k = 0; k < initialBiasWeights.get(i).get(j).size(); k++) {
                    weights2.add(initialBiasWeights.get(i).get(j).get(k));
                }
                weights1.add(weights2);
            }
            biasWeights.add(weights1);
        }

        return biasWeights;

    }

    /*===========================================================================
     * Copy the initial weights of the network
     *===========================================================================*/
    static ArrayList<ArrayList<Double[][]>> copyWeights(ArrayList<ArrayList<Double[][]>> initialNeuralNet,
            ArrayList<Integer[]> arrayArchitecture) {

        ArrayList<ArrayList<Double[][]>> neuralNet = initializeWeights(arrayArchitecture);

        for (int i = 0; i < initialNeuralNet.size(); i++) {
            for (int j = 0; j < initialNeuralNet.get(i).size(); j++) {
                for (int k = 0; k < initialNeuralNet.get(i).get(j).length; k++) {
                    for (int l = 0; l < initialNeuralNet.get(i).get(j)[k].length; l++) {
                        neuralNet.get(i).get(j)[k][l] = initialNeuralNet.get(i).get(j)[k][l];
                    }
                }
            }
        }

        return neuralNet;
    }

    /*===========================================================================
     * Initialize the weights of the biases
     *===========================================================================*/
    static ArrayList<ArrayList<ArrayList<Double>>> initializeBiasWeights(ArrayList<Integer[]> arrayArchitecture) {

        ArrayList<ArrayList<ArrayList<Double>>> biasWeights = new ArrayList<ArrayList<ArrayList<Double>>>();
        ArrayList<ArrayList<Double>> biasWeights1 = new ArrayList<ArrayList<Double>>();
        ArrayList<ArrayList<Double>> biasWeights2 = new ArrayList<ArrayList<Double>>();
        Random randomGenerator = new Random();

        for (int i = 0; i < (arrayArchitecture.get(1).length - 1); i++) {

            int numNeurons = arrayArchitecture.get(1)[i];
            ArrayList<Double> weights = new ArrayList<Double>();

            for (int j = 0; j < numNeurons; j++) {

                //double num = 2 * randomGenerator.nextDouble() - 1;
                double num = 0.2 * randomGenerator.nextDouble() - 0.1;
                weights.add(num);
            }

            biasWeights1.add(weights);
        }

        for (int i = 1; i < (arrayArchitecture.get(0).length); i++) {
            int numNeurons = 0;
            //if(i < (arrayArchitecture.get(0).length-1)){
            numNeurons = arrayArchitecture.get(0)[i] - arrayArchitecture.get(0)[0]; //Aqui utiliza exemplos+classes
            //numNeurons = arrayArchitecture.get(0)[i]; //Aqui utiliza so exemplos
            //}else{
            //numNeurons = arrayArchitecture.get(0)[i];
            //}
            ArrayList<Double> weights = new ArrayList<Double>();

            for (int j = 0; j < numNeurons; j++) {

                //double num = 2 * randomGenerator.nextDouble() - 1;
                double num = 0.2 * randomGenerator.nextDouble() - 0.1;
                weights.add(num);
            }

            biasWeights2.add(weights);
        }

        biasWeights.add(biasWeights1);
        biasWeights.add(biasWeights2);

        return biasWeights;
    }

    /*===========================================================================
     * Get the initial bias weights of the network
     *===========================================================================*/
    static ArrayList<ArrayList<ArrayList<Double>>> getBiasWeights(ArrayList<Integer[]> arrayArchitecture, ArrayList<ArrayList<ArrayList<Double>>> biasWeights) {

        ArrayList<ArrayList<ArrayList<Double>>> biasVariationsPrevious = new ArrayList<ArrayList<ArrayList<Double>>>();

        for (int i = 0; i < biasWeights.size(); i++) {
            ArrayList<ArrayList<Double>> weights1 = new ArrayList<ArrayList<Double>>();
            for (int j = 0; j < biasWeights.get(i).size(); j++) {
                ArrayList<Double> weights2 = new ArrayList<Double>();
                for (int k = 0; k < biasWeights.get(i).get(j).size(); k++) {
                    weights2.add(biasWeights.get(i).get(j).get(k));
                }
                weights1.add(weights2);
            }
            biasVariationsPrevious.add(weights1);
        }

        return biasVariationsPrevious;
    }

    /*===========================================================================
     * Build structure to store the outputs of the neural network, in each layer.
     * Each output layer represents a level of the hierarchy, if the layer is not
     * a hidden layer
     *===========================================================================*/
    static ArrayList<ArrayList<Double[]>> buildOutputStructure(ArrayList<Integer[]> arrayArchitecture,
            ArrayList<ArrayList<Integer>> indexesLevels) {

        ArrayList<ArrayList<Double[]>> outputs = new ArrayList<ArrayList<Double[]>>();
        ArrayList<Double[]> outputs1 = new ArrayList<Double[]>();
        ArrayList<Double[]> outputs2 = new ArrayList<Double[]>();

        for (int i = 0; i < arrayArchitecture.get(1).length - 1; i++) {

            Double output[];
            Double output2[];
            output = new Double[arrayArchitecture.get(1)[i]];
            //if(i < (arrayArchitecture.get(1).length-2)){
            output2 = new Double[arrayArchitecture.get(0)[i + 1] - arrayArchitecture.get(0)[0]]; //Aqui utiliza exemplos+classes
            //output2 = new Double[indexesLevels.get(i).size()]; //Aqui utiliza so exemplos
            //}else{
            //    output2 = new Double[arrayArchitecture.get(0)[i+1]];
            // }

            for (int j = 0; j < output.length; j++) {
                output[j] = 0.0;
            }
            for (int j = 0; j < output2.length; j++) {
                output2[j] = 0.0;
            }
            outputs1.add(output);
            outputs2.add(output2);
        }

        outputs.add(outputs1);
        outputs.add(outputs2);

        return outputs;
    }

    /*===========================================================================
     * Calculate the output resulted from the data input and the first hidden layer
     *===========================================================================*/
    //Use logistic function
    static void outputsDataInputFirstHLayer(ArrayList<Integer[]> arrayArchitecture,
            ArrayList<ArrayList<Double>> datasetTrain,
            ArrayList<ArrayList<Double[][]>> neuralNet,
            ArrayList<ArrayList<Double[]>> outputs, int numInInst, double a,
            ArrayList<ArrayList<ArrayList<Double>>> biasWeights) {

        for (int j = 0; j < arrayArchitecture.get(1)[0]; j++) {
            for (int i = 0; i < arrayArchitecture.get(0)[0]; i++) {
                outputs.get(0).get(0)[j] += datasetTrain.get(numInInst).get(i) * neuralNet.get(0).get(0)[i][j];
            }
            outputs.get(0).get(0)[j] += 1 * biasWeights.get(0).get(0).get(j);
            outputs.get(0).get(0)[j] = sigmoidalLogistic(outputs.get(0).get(0)[j], a);
        }
    }
    //Use hiperbolic tangent function
    static void outputsDataInputFirstHLayer(ArrayList<Integer[]> arrayArchitecture,
            ArrayList<ArrayList<Double>> datasetTrain,
            ArrayList<ArrayList<Double[][]>> neuralNet,
            ArrayList<ArrayList<Double[]>> outputs, int numInInst, double b, double c,
            ArrayList<ArrayList<ArrayList<Double>>> biasWeights) {

        for (int j = 0; j < arrayArchitecture.get(1)[0]; j++) {
            for (int i = 0; i < arrayArchitecture.get(0)[0]; i++) {
                outputs.get(0).get(0)[j] += datasetTrain.get(numInInst).get(i) * neuralNet.get(0).get(0)[i][j];
            }
            outputs.get(0).get(0)[j] += 1 * biasWeights.get(0).get(0).get(j);
            outputs.get(0).get(0)[j] = sigmoidalHiperbolicTangent(outputs.get(0).get(0)[j], b, c);
        }
    }

    /*===========================================================================
     * Calculate the output resulted from the first hidden layer and the first level
     *===========================================================================*/
    //Logistic Function
    static double outputsFirstHLayerFirstLevel(ArrayList<Integer[]> arrayArchitecture,
            ArrayList<ArrayList<Double[][]>> neuralNet,
            ArrayList<ArrayList<Double[]>> outputs, int numInInst, double a,
            ArrayList<ArrayList<ArrayList<Double>>> biasWeights) {

        double sumWeights = 0;

        for (int j = 0; j < (arrayArchitecture.get(0)[1] - arrayArchitecture.get(0)[0]); j++) { //Aqui utiliza exemplos+classes
        //for (int j = 0; j < arrayArchitecture.get(0)[1]; j++) { //Aqui utiliza so exemplos
            for (int i = 0; i < outputs.get(0).get(0).length; i++) {
                outputs.get(1).get(0)[j] += outputs.get(0).get(0)[i] * neuralNet.get(1).get(0)[i][j];

                sumWeights += (Math.pow(neuralNet.get(1).get(0)[i][j], 2) / (1 + Math.pow(neuralNet.get(1).get(0)[i][j], 2)));

            }
            outputs.get(1).get(0)[j] += 1 * biasWeights.get(1).get(0).get(j);
            outputs.get(1).get(0)[j] = sigmoidalLogistic(outputs.get(1).get(0)[j], a);
        }

        return sumWeights;
    }

    //Hiperbolic tangent Function
    static double outputsFirstHLayerFirstLevel(ArrayList<Integer[]> arrayArchitecture,
            ArrayList<ArrayList<Double[][]>> neuralNet,
            ArrayList<ArrayList<Double[]>> outputs, int numInInst, double b, double c,
            ArrayList<ArrayList<ArrayList<Double>>> biasWeights) {

        double sumWeights = 0;

        for (int j = 0; j < (arrayArchitecture.get(0)[1] - arrayArchitecture.get(0)[0]); j++) {
            for (int i = 0; i < outputs.get(0).get(0).length; i++) {
                outputs.get(1).get(0)[j] += outputs.get(0).get(0)[i] * neuralNet.get(1).get(0)[i][j];

                sumWeights += (Math.pow(neuralNet.get(1).get(0)[i][j], 2) / (1 + Math.pow(neuralNet.get(1).get(0)[i][j], 2)));

            }
            outputs.get(1).get(0)[j] += 1 * biasWeights.get(1).get(0).get(j);
            outputs.get(1).get(0)[j] = sigmoidalHiperbolicTangent(outputs.get(1).get(0)[j], b, c);
        }

        return sumWeights;
    }

    /*===========================================================================
     * Sigmoidal logistic function
     *===========================================================================*/
    static double sigmoidalLogistic(double inputValue, double a) {

        double output = 1 / (1 + Math.exp(-a * inputValue));

        return output;
    }

    /*===========================================================================
     * Sigmoidal hiperbolic tangent function
     *===========================================================================*/
    static double sigmoidalHiperbolicTangent(double inputValue, double b, double c) {

        double output = b * Math.tanh(c * inputValue);

        return output;
    }

    /*===========================================================================
     * Local gradient for output neuron and square error over the outputs
     *===========================================================================*/
    //Logistic function
    static double localGradientOutputNeuron(ArrayList<Double> localGradientOutput,
            ArrayList<ArrayList<Double[]>> outputs,
            ArrayList<ArrayList<Integer>> indexesLevels,
            ArrayList<ArrayList<Double>> datasetTrain, int numInInst,
            double a, int param, double weightDecay, double sumWeights) {

        double squareError = 0;

        for (int k = 0; k < outputs.get(1).get(0).length; k++) {
            int indexClass = indexesLevels.get(0).get(k);
            double error = 0;

            switch (param) {
                case 1:
                    error = conventionalError(datasetTrain.get(numInInst).get(indexClass), outputs.get(1).get(0)[k]);
                    break;
                case 2:
                    error = FunctionsMultiLabel.errorZhangZhou(datasetTrain.get(numInInst), outputs.get(1).get(0), indexesLevels.get(0), indexClass, k);
                    break;
            }

            squareError += Math.pow(error, 2);

            localGradientOutput.add(error * a * outputs.get(1).get(0)[k] * (1 - outputs.get(1).get(0)[k]));
        }

        squareError += 0.5 * weightDecay * sumWeights;

        return squareError;
    }

    //Hiperbolic tangent function
    static double localGradientOutputNeuron(ArrayList<Double> localGradientOutput,
            ArrayList<ArrayList<Double[]>> outputs,
            ArrayList<ArrayList<Integer>> indexesLevels,
            ArrayList<ArrayList<Double>> datasetTrain, int numInInst,
            double b, double c, int param, double weightDecay, double sumWeights) {

        double squareError = 0;

        for (int k = 0; k < outputs.get(1).get(0).length; k++) {
            int indexClass = indexesLevels.get(0).get(k);
            double error = 0;

            switch (param) {
                case 1:
                    error = conventionalError(datasetTrain.get(numInInst).get(indexClass), outputs.get(1).get(0)[k]);
                    break;
                case 2:
                    error = FunctionsMultiLabel.errorZhangZhou(datasetTrain.get(numInInst), outputs.get(1).get(0), indexesLevels.get(0), indexClass, k);
                    break;
            }

            squareError += Math.pow(error, 2);
            localGradientOutput.add((c / b) * error * (b - outputs.get(1).get(0)[k]) * (b + outputs.get(1).get(0)[k]));

        }

        squareError += 0.5 * weightDecay * sumWeights;

        return squareError;
    }

    /*===========================================================================
     * Conventional error function
     *===========================================================================*/
    static double conventionalError(double realOutput, double predictedOutput) {

        double error = realOutput - predictedOutput;

        return error;
    }

    /*===========================================================================
     * Update the weights between the first hidden layer and the first level layer
     *===========================================================================*/
    static void updateWeightsFHLayerFLLayer(ArrayList<Integer[]> arrayArchitecture,
            ArrayList<ArrayList<Double[]>> outputs,
            ArrayList<Double> localGradientOutput,
            ArrayList<ArrayList<Double[][]>> neuralNet,
            ArrayList<ArrayList<Double[][]>> variationsPrevious,
            double learningRate, double momentumConstant,
            int numInInst, double weightDecay,
            ArrayList<ArrayList<ArrayList<Double>>> biasWeights,
            ArrayList<ArrayList<ArrayList<Double>>> biasVariationsPrevious) {

        for (int j = 0; j < arrayArchitecture.get(0)[1] - arrayArchitecture.get(0)[0]; j++) {//Aqui utiliza exemplos+classes
        //for (int j = 0; j < arrayArchitecture.get(0)[1]; j++) {//Aqui utiliza so exemplos
            double delta = 0;
            double deltaBias = learningRate * localGradientOutput.get(j) * 1;

            //Update bias weights
            if (numInInst == 0) {
                double bias = biasWeights.get(1).get(0).get(j) + deltaBias;
                biasWeights.get(1).get(0).set(j, bias);
                biasVariationsPrevious.get(1).get(0).set(j, deltaBias);
            } else {
                double bias = biasWeights.get(1).get(0).get(j) + momentumConstant * biasVariationsPrevious.get(1).get(0).get(j) + deltaBias;
                biasWeights.get(1).get(0).set(j, bias);
                biasVariationsPrevious.get(1).get(0).set(j, deltaBias);
            }

            for (int i = 0; i < outputs.get(0).get(0).length; i++) {

                double weightDecayContrib = learningRate * weightDecay * (neuralNet.get(1).get(0)[i][j] / (Math.pow(1 + Math.pow(neuralNet.get(1).get(0)[i][j], 2), 2)));

                delta = learningRate * localGradientOutput.get(j) * outputs.get(0).get(0)[i] - weightDecayContrib;

                if (numInInst == 0) {
                    neuralNet.get(1).get(0)[i][j] = neuralNet.get(1).get(0)[i][j] + delta;
                    variationsPrevious.get(1).get(0)[i][j] = delta;
                } else {
                    neuralNet.get(1).get(0)[i][j] = neuralNet.get(1).get(0)[i][j] + momentumConstant * variationsPrevious.get(1).get(0)[i][j] + delta;
                    variationsPrevious.get(1).get(0)[i][j] = delta;
                }
            }
        }
    }

    /*===========================================================================
     * Local gradient for hidden neuron
     *===========================================================================*/
    //Use logistic function
    static void localGradientHiddenNeuron(ArrayList<Double> localGradientHidden,
            ArrayList<Double> localGradientOutput,
            ArrayList<ArrayList<Double[]>> outputs,
            ArrayList<ArrayList<Double[][]>> neuralNet, double a) {

        for (int i = 0; i < outputs.get(0).get(0).length; i++) {
            double sumlocalGradientOutput = 0;
            for (int j = 0; j < outputs.get(1).get(0).length; j++) {
                sumlocalGradientOutput += (localGradientOutput.get(j) * neuralNet.get(1).get(0)[i][j]);
            }
            localGradientHidden.add(a * outputs.get(0).get(0)[i] * (1 - outputs.get(0).get(0)[i]) * sumlocalGradientOutput);
        }
    }

    static void localGradientHiddenNeuron(ArrayList<Double> localGradientHidden,
            ArrayList<Double> localGradientOutput,
            ArrayList<ArrayList<Double[]>> outputs,
            ArrayList<ArrayList<Double[][]>> neuralNet, double a, int level) {

        for (int i = 0; i < outputs.get(0).get(level).length; i++) {
            double sumlocalGradientOutput = 0;
            for (int j = 0; j < outputs.get(1).get(level).length; j++) {
                sumlocalGradientOutput += (localGradientOutput.get(j) * neuralNet.get(1).get(level)[i][j]);
            }
            localGradientHidden.add(a * outputs.get(0).get(level)[i] * (1 - outputs.get(0).get(level)[i]) * sumlocalGradientOutput);
        }
    }

    //Use hiperbolic tangent function
    static void localGradientHiddenNeuron(ArrayList<Double> localGradientHidden,
            ArrayList<Double> localGradientOutput,
            ArrayList<ArrayList<Double[]>> outputs,
            ArrayList<ArrayList<Double[][]>> neuralNet, double b, double c) {

        for (int i = 0; i < outputs.get(0).get(0).length; i++) {
            double sumlocalGradientOutput = 0;
            for (int j = 0; j < outputs.get(1).get(0).length; j++) {
                sumlocalGradientOutput += (localGradientOutput.get(j) * neuralNet.get(1).get(0)[i][j]);
            }
            localGradientHidden.add((c / b) * (b - outputs.get(0).get(0)[i]) * (b + outputs.get(0).get(0)[i]) * sumlocalGradientOutput);
        }
    }

    static void localGradientHiddenNeuron(ArrayList<Double> localGradientHidden,
            ArrayList<Double> localGradientOutput,
            ArrayList<ArrayList<Double[]>> outputs,
            ArrayList<ArrayList<Double[][]>> neuralNet, double b, double c, int level) {

        for (int i = 0; i < outputs.get(0).get(level).length; i++) {
            double sumlocalGradientOutput = 0;
            for (int j = 0; j < outputs.get(1).get(level).length; j++) {
                sumlocalGradientOutput += (localGradientOutput.get(j) * neuralNet.get(1).get(level)[i][j]);
            }
            localGradientHidden.add((c / b) * (b - outputs.get(0).get(level)[i]) * (b + outputs.get(0).get(level)[i]) * sumlocalGradientOutput);
        }
    }

    /*===========================================================================
     * Update the weights between the input layer and the first hidden layer
     *===========================================================================*/
    static void updateWeightsFILayerFHLayer(ArrayList<Integer[]> arrayArchitecture,
            ArrayList<Double> localGradientHidden,
            ArrayList<ArrayList<Double>> datasetTrain,
            ArrayList<ArrayList<Double[][]>> neuralNet,
            ArrayList<ArrayList<Double[][]>> variationsPrevious,
            double learningRate, double momentumConstant, int numInInst,
            ArrayList<ArrayList<ArrayList<Double>>> biasWeights,
            ArrayList<ArrayList<ArrayList<Double>>> biasVariationsPrevious,
            double weightDecay) {

        for (int j = 0; j < arrayArchitecture.get(1)[0]; j++) {
            double delta = 0;
            double deltaBias = learningRate * localGradientHidden.get(j) * 1;

            //Update bias weights
            if (numInInst == 0) {
                double bias = biasWeights.get(0).get(0).get(j) + deltaBias;
                biasWeights.get(0).get(0).set(j, bias);
                biasVariationsPrevious.get(0).get(0).set(j, deltaBias);
            } else {
                double bias = biasWeights.get(0).get(0).get(j) + momentumConstant * biasVariationsPrevious.get(0).get(0).get(j) + deltaBias;
                biasWeights.get(0).get(0).set(j, bias);
                biasVariationsPrevious.get(0).get(0).set(j, deltaBias);
            }

            //Update neural net weights
            for (int i = 0; i < arrayArchitecture.get(0)[0]; i++) {

                double weightDecayContrib = learningRate * weightDecay * (neuralNet.get(0).get(0)[i][j] / (Math.pow(1 + Math.pow(neuralNet.get(0).get(0)[i][j], 2), 2)));

                delta = learningRate * localGradientHidden.get(j) * datasetTrain.get(numInInst).get(i) - weightDecayContrib;

                if (numInInst == 0) {
                    neuralNet.get(0).get(0)[i][j] = neuralNet.get(0).get(0)[i][j] + delta;
                    variationsPrevious.get(0).get(0)[i][j] = delta;
                } else {
                    neuralNet.get(0).get(0)[i][j] = neuralNet.get(0).get(0)[i][j] + momentumConstant * variationsPrevious.get(0).get(0)[i][j] + delta;
                    variationsPrevious.get(0).get(0)[i][j] = delta;
                }
            }
        }
    }

    /*===========================================================================
     * Calculate the output resulted from one level and one hidden layer
     *===========================================================================*/
    //Use logistic function
    static void outputsOneLevelOneHiddenLayer(ArrayList<Integer[]> arrayArchitecture,
            ArrayList<ArrayList<Double[]>> outputs,
            ArrayList<ArrayList<Double[][]>> neuralNet, int actLevel, double a,
            ArrayList<ArrayList<ArrayList<Double>>> biasWeights,
            ArrayList<ArrayList<Double>> datasetTrain,
            int numInInst,
            ArrayList<ArrayList<Integer>> indexesLevels) {

        for (int j = 0; j < arrayArchitecture.get(1)[actLevel]; j++) {
            for (int i = 0; i < arrayArchitecture.get(0)[actLevel]; i++) { //Aqui utiliza exemplos+classes
            //for (int i = 0; i < arrayArchitecture.get(0)[0]; i++) { //Aqui utiliza so exemplos
                //IF-ELSE descomentados sao para usar exemplos+classes
                if (i < arrayArchitecture.get(0)[0]) {
                    outputs.get(0).get(actLevel)[j] += datasetTrain.get(numInInst).get(i) * neuralNet.get(0).get(actLevel)[i][j];

                } else {
                    //Use predicted classes as feature vectors
                    outputs.get(0).get(actLevel)[j] += outputs.get(1).get(actLevel - 1)[i - arrayArchitecture.get(0)[0]] * neuralNet.get(0).get(actLevel)[i][j];
                    
                    //Use true classes as feature vectors
                    //int indexClass = indexesLevels.get(actLevel-1).get(i - arrayArchitecture.get(0)[0]);
                    //outputs.get(0).get(actLevel)[j] += datasetTrain.get(numInInst).get(indexClass) * neuralNet.get(0).get(actLevel)[i][j];
                }
            }
            outputs.get(0).get(actLevel)[j] += 1 * biasWeights.get(0).get(actLevel).get(j);
            outputs.get(0).get(actLevel)[j] = sigmoidalLogistic(outputs.get(0).get(actLevel)[j], a);
        }
    }
    
    //Use logistic function
    static void outputsOneLevelOneHiddenLayer(ArrayList<Integer[]> arrayArchitecture,
            ArrayList<ArrayList<Double[]>> outputs,
            ArrayList<ArrayList<Double[][]>> neuralNet, int actLevel, double a,
            ArrayList<ArrayList<ArrayList<Double>>> biasWeights,
            ArrayList<ArrayList<Double>> datasetTrain,
            int numInInst) {

        for (int j = 0; j < arrayArchitecture.get(1)[actLevel]; j++) {
            for (int i = 0; i < arrayArchitecture.get(0)[actLevel]; i++) {//Aqui utiliza exemplos+classes
            //for (int i = 0; i < arrayArchitecture.get(0)[0]; i++) {//Aqui utiliza so exemplos
                //IF-ELSE descomentados sao para usar exemplos+classes
                if (i < arrayArchitecture.get(0)[0]) {
                    outputs.get(0).get(actLevel)[j] += datasetTrain.get(numInInst).get(i) * neuralNet.get(0).get(actLevel)[i][j];
                } else {
                    //Use predicted classes as feature vectors
                    outputs.get(0).get(actLevel)[j] += outputs.get(1).get(actLevel - 1)[i - arrayArchitecture.get(0)[0]] * neuralNet.get(0).get(actLevel)[i][j];
                }
            }
            outputs.get(0).get(actLevel)[j] += 1 * biasWeights.get(0).get(actLevel).get(j);
            outputs.get(0).get(actLevel)[j] = sigmoidalLogistic(outputs.get(0).get(actLevel)[j], a);
        }
    }
    
     //Use logistic function
    static void outputsOneLevelOneHiddenLayer(ArrayList<Integer[]> arrayArchitecture,
            ArrayList<ArrayList<Double[]>> outputs,
            ArrayList<ArrayList<Double[][]>> neuralNet, int actLevel, double a,
            ArrayList<ArrayList<ArrayList<Double>>> biasWeights,
            ArrayList<ArrayList<Double>> datasetTrain,
            int numInInst,
            double threshold,
            double thresholdReductionFactor) {

        for (int j = 0; j < arrayArchitecture.get(1)[actLevel]; j++) {
            for (int i = 0; i < arrayArchitecture.get(0)[actLevel]; i++) {

                if (i < arrayArchitecture.get(0)[0]) {
                    outputs.get(0).get(actLevel)[j] += datasetTrain.get(numInInst).get(i) * neuralNet.get(0).get(actLevel)[i][j];
                } else {
                    //Use predicted classes as feature vectors, but converted to 0 or 1
                    double binaryOutput = FunctionsMultiLabel.getOutputThreshold(outputs.get(1).get(actLevel - 1)[i - arrayArchitecture.get(0)[0]], threshold, thresholdReductionFactor, actLevel);
                    outputs.get(0).get(actLevel)[j] += binaryOutput * neuralNet.get(0).get(actLevel)[i][j];
                }
            }
            outputs.get(0).get(actLevel)[j] += 1 * biasWeights.get(0).get(actLevel).get(j);
            outputs.get(0).get(actLevel)[j] = sigmoidalLogistic(outputs.get(0).get(actLevel)[j], a);
        }
    }

    //Use hiperbolic tangent function
    static void outputsOneLevelOneHiddenLayer(ArrayList<Integer[]> arrayArchitecture,
            ArrayList<ArrayList<Double[]>> outputs,
            ArrayList<ArrayList<Double[][]>> neuralNet, int actLevel, double b, double c,
            ArrayList<ArrayList<ArrayList<Double>>> biasWeights,
            ArrayList<ArrayList<Double>> datasetTrain,
            int numInInst) {

        for (int j = 0; j < arrayArchitecture.get(1)[actLevel]; j++) {
            for (int i = 0; i < arrayArchitecture.get(0)[actLevel]; i++) {

                if (i < arrayArchitecture.get(0)[0]) {
                    outputs.get(0).get(actLevel)[j] += datasetTrain.get(numInInst).get(i) * neuralNet.get(0).get(actLevel)[i][j];
                } else {
                    //Uncomment to use classes as feature vectors
                    outputs.get(0).get(actLevel)[j] += outputs.get(1).get(actLevel - 1)[i - arrayArchitecture.get(0)[0]] * neuralNet.get(0).get(actLevel)[i][j];
                }
            }
            outputs.get(0).get(actLevel)[j] += 1 * biasWeights.get(0).get(actLevel).get(j);
            outputs.get(0).get(actLevel)[j] = sigmoidalHiperbolicTangent(outputs.get(0).get(actLevel)[j], b, c);
        }
    }

    /*===========================================================================
     * Calculate the output resulted from hidden layer and one level
     *===========================================================================*/
    //Logistic function
    static double outputsOneHiddenLayerOneLevel(ArrayList<Integer[]> arrayArchitecture,
            ArrayList<ArrayList<Double[]>> outputs,
            ArrayList<ArrayList<Double[][]>> neuralNet, int actLevel, double a,
            ArrayList<ArrayList<ArrayList<Double>>> biasWeights) {

        double sumWeights = 0;
        
        for (int j = 0; j < (arrayArchitecture.get(0)[actLevel + 1] - arrayArchitecture.get(0)[0]); j++) { //Aqui utiliza exemplos+classes
        //for (int j = 0; j < arrayArchitecture.get(0)[actLevel + 1]; j++) {//Aqui utiliza so exemplos
            for (int i = 0; i < outputs.get(0).get(actLevel).length; i++) {
                outputs.get(1).get(actLevel)[j] += outputs.get(0).get(actLevel)[i] * neuralNet.get(1).get(actLevel)[i][j];

                sumWeights += (Math.pow(neuralNet.get(1).get(actLevel)[i][j], 2) / (1 + Math.pow(neuralNet.get(1).get(actLevel)[i][j], 2)));

            }
            outputs.get(1).get(actLevel)[j] += 1 * biasWeights.get(1).get(actLevel).get(j);
            outputs.get(1).get(actLevel)[j] = sigmoidalLogistic(outputs.get(1).get(actLevel)[j], a);
        }

        return sumWeights;
    }

    //Hiperbolic tangent function
    static double outputsOneHiddenLayerOneLevel(ArrayList<Integer[]> arrayArchitecture,
            ArrayList<ArrayList<Double[]>> outputs,
            ArrayList<ArrayList<Double[][]>> neuralNet, int actLevel, double b, double c,
            ArrayList<ArrayList<ArrayList<Double>>> biasWeights) {

        double sumWeights = 0;

        for (int j = 0; j < (arrayArchitecture.get(0)[actLevel + 1] - arrayArchitecture.get(0)[0]); j++) {
            for (int i = 0; i < outputs.get(0).get(actLevel).length; i++) {
                outputs.get(1).get(actLevel)[j] += outputs.get(0).get(actLevel)[i] * neuralNet.get(1).get(actLevel)[i][j];

                sumWeights += (Math.pow(neuralNet.get(1).get(actLevel)[i][j], 2) / (1 + Math.pow(neuralNet.get(1).get(actLevel)[i][j], 2)));

            }
            outputs.get(1).get(actLevel)[j] += 1 * biasWeights.get(1).get(actLevel).get(j);
            outputs.get(1).get(actLevel)[j] = sigmoidalHiperbolicTangent(outputs.get(1).get(actLevel)[j], b, c);
        }

        return sumWeights;
    }

    /*===========================================================================
     * Compute the error between the real classes and prediction of the network
     *===========================================================================*/
    static double errorRealPredicted(ArrayList<ArrayList<Double[]>> outputs,
            ArrayList<ArrayList<Integer>> indexesLevels,
            ArrayList<ArrayList<Double>> datasetTrain,
            ArrayList<Double> outputErrors, int level,
            int numInInst, int param, double weightDecay, double sumWeights) {

        //Square error over the outputs
        double squareError = 0;

        for (int i = 0; i < outputs.get(1).get(level).length; i++) {
            int indexClass = indexesLevels.get(level).get(i);
            double error = 0;

            switch (param) {
                case 1:
                    error = conventionalError(datasetTrain.get(numInInst).get(indexClass), outputs.get(1).get(level)[i]);
                    break;
                case 2:
                    error = FunctionsMultiLabel.errorZhangZhou(datasetTrain.get(numInInst), outputs.get(1).get(level),
                            indexesLevels.get(level), indexClass, i);
                    break;
            }

            squareError += Math.pow(error, 2);
            outputErrors.add(error);
        }

        squareError += 0.5 * weightDecay * sumWeights;

        return squareError;
    }

    /*===========================================================================
     * Local gradient for a given output neuron
     *===========================================================================*/
    //Logistic function
    static void localGradientOutputActualNeuron(ArrayList<ArrayList<Double[]>> outputs,
            ArrayList<Double> localGradientOutput,
            ArrayList<Double> outputErrors, int level, double a) {

        for (int k = 0; k < outputs.get(1).get(level).length; k++) {
            localGradientOutput.add(outputErrors.get(k) * a * outputs.get(1).get(level)[k] * (1 - outputs.get(1).get(level)[k]));
        }
    }
    //Hiperbolic tangent function
    static void localGradientOutputActualNeuron(ArrayList<ArrayList<Double[]>> outputs,
            ArrayList<Double> localGradientOutput,
            ArrayList<Double> outputErrors, int level, double b, double c) {

        for (int k = 0; k < outputs.get(1).get(level).length; k++) {
            localGradientOutput.add((c / b) * outputErrors.get(k) * (b - outputs.get(1).get(level)[k]) * (b + outputs.get(1).get(level)[k]));
        }
    }

    /*===========================================================================
     * Update the weights between an hidden layer and a level layer - BP
     *===========================================================================*/
    static void updateWeightsHiddenLayerLevelLayer(ArrayList<Integer[]> arrayArchitecture,
            ArrayList<ArrayList<Double[]>> outputs,
            ArrayList<Double> localGradientOutput,
            ArrayList<ArrayList<Double[][]>> neuralNet,
            ArrayList<ArrayList<Double[][]>> variationsPrevious,
            double learningRate, double momentumConstant,
            int level, int numInInst, double weightDecay,
            ArrayList<ArrayList<ArrayList<Double>>> biasWeights,
            ArrayList<ArrayList<ArrayList<Double>>> biasVariationsPrevious) {

        for (int j = 0; j < (arrayArchitecture.get(0)[level + 1] - arrayArchitecture.get(0)[0]); j++) {//Aqui utiliza exemplos+classes
        //for (int j = 0; j < arrayArchitecture.get(0)[level + 1]; j++) {//Aqui utiliza so exemplos
            double delta = 0;
            double deltaBias = learningRate * localGradientOutput.get(j) * 1;

            //Update bias weights
            if (numInInst == 0) {
                double bias = biasWeights.get(1).get(level).get(j) + deltaBias;
                biasWeights.get(1).get(level).set(j, bias);
                biasVariationsPrevious.get(1).get(level).set(j, deltaBias);
            } else {
                double bias = biasWeights.get(1).get(level).get(j) + momentumConstant * biasVariationsPrevious.get(1).get(level).get(j) + deltaBias;
                biasWeights.get(1).get(level).set(j, bias);
                biasVariationsPrevious.get(1).get(level).set(j, deltaBias);
            }

            for (int i = 0; i < outputs.get(0).get(level).length; i++) {

                double weightDecayContrib = learningRate * weightDecay * (neuralNet.get(1).get(level)[i][j] / (Math.pow(1 + Math.pow(neuralNet.get(1).get(level)[i][j], 2), 2)));

                delta = learningRate * localGradientOutput.get(j) * outputs.get(0).get(level)[i] - weightDecayContrib;

                if (numInInst == 0) {
                    neuralNet.get(1).get(level)[i][j] = neuralNet.get(1).get(level)[i][j] + delta;
                    variationsPrevious.get(1).get(level)[i][j] = delta;
                } else {
                    neuralNet.get(1).get(level)[i][j] = neuralNet.get(1).get(level)[i][j] + momentumConstant * variationsPrevious.get(1).get(level)[i][j] + delta;
                    variationsPrevious.get(1).get(level)[i][j] = delta;
                }
            }
        }
    }


    /*===========================================================================
     * Update the weights between an hidden layer and a level layer - Rprop
     *===========================================================================*/
    static void updateWeightsHiddenLayerLevelLayer(ArrayList<Integer[]> arrayArchitecture,
            ArrayList<ArrayList<Double[][]>> gradients,
            ArrayList<ArrayList<Double[][]>> deltas,
            ArrayList<ArrayList<Double[][]>> deltasPrevious,
            ArrayList<ArrayList<Double[][]>> gradientsPrevious,
            ArrayList<ArrayList<Double[][]>> neuralNet,
            ArrayList<ArrayList<Double[][]>> variationsPrevious,
            ArrayList<ArrayList<Double[]>> outputs,
            double increaseFactor,
            double decreaseFactor,
            double initialDelta,
            double maxDelta,
            double minDelta,
            int level,
            ArrayList<ArrayList<ArrayList<Double>>> gradientsBias,
            ArrayList<ArrayList<ArrayList<Double>>> gradientsBiasPrevious,
            ArrayList<ArrayList<ArrayList<Double>>> biasWeights,
            ArrayList<ArrayList<ArrayList<Double>>> biasVariationsPrevious,
            ArrayList<ArrayList<ArrayList<Double>>> deltasBias,
            double currentError,
            double previousError) {

        for (int j = 0; j < arrayArchitecture.get(0)[level + 1] - arrayArchitecture.get(0)[0]; j++) {//Aqui utiliza exemplos+classes
        //for (int j = 0; j < arrayArchitecture.get(0)[level + 1]; j++) {//Aqui utiliza so exemplos

            if ((gradientsBiasPrevious.get(1).get(level).get(j) * gradientsBias.get(1).get(level).get(j)) > 0) {
                deltasBias.get(1).get(level).set(j, Math.min(deltasBias.get(1).get(level).get(j) * increaseFactor, maxDelta));
                biasVariationsPrevious.get(1).get(level).set(j, -(Math.signum(gradientsBias.get(1).get(level).get(j)) * deltasBias.get(1).get(level).get(j)));
                biasWeights.get(1).get(level).set(j, biasWeights.get(1).get(level).get(j) + biasVariationsPrevious.get(1).get(level).get(j));
                gradientsBiasPrevious.get(1).get(level).set(j, gradientsBias.get(1).get(level).get(j));

            } else if ((gradientsBiasPrevious.get(1).get(level).get(j) * gradientsBias.get(1).get(level).get(j)) < 0) {
                deltasBias.get(1).get(level).set(j, Math.max(deltasBias.get(1).get(level).get(j) * decreaseFactor, minDelta));

                //Comment the IF conditional below to use the Rprop+
                //If not commented, the algorithm used is the iRprop+
                //iRprop+ was proposed by Christian Igel and Michael Husken - Improving the Rprop Learning Algorithm, 2000
                //if(currentError > previousError){
                biasWeights.get(1).get(level).set(j, biasWeights.get(1).get(level).get(j) - biasVariationsPrevious.get(1).get(level).get(j));
                //}    
                //gradientsBiasPrevious.get(1).get(level + 1).set(j, gradientsBias.get(1).get(level + 1).get(j));
                gradientsBiasPrevious.get(1).get(level).set(j, 0.0);

            } else if ((gradientsBiasPrevious.get(1).get(level).get(j) * gradientsBias.get(1).get(level).get(j)) == 0) {
                biasVariationsPrevious.get(1).get(level).set(j, -(Math.signum(gradientsBias.get(1).get(level).get(j)) * deltasBias.get(1).get(level).get(j)));
                biasWeights.get(1).get(level).set(j, biasWeights.get(1).get(level).get(j) + biasVariationsPrevious.get(1).get(level).get(j));
                gradientsBiasPrevious.get(1).get(level).set(j, gradientsBias.get(1).get(level).get(j));
            }

            for (int i = 0; i < outputs.get(0).get(level).length; i++) {

                //System.out.println("Gradients Previous = " + gradientsPrevious.get(1).get(level)[i][j]);

                if ((gradientsPrevious.get(1).get(level)[i][j] * gradients.get(1).get(level)[i][j]) > 0) {
                    deltas.get(1).get(level)[i][j] = Math.min(deltas.get(1).get(level)[i][j] * increaseFactor, maxDelta);
                    variationsPrevious.get(1).get(level)[i][j] = -(Math.signum(gradients.get(1).get(level)[i][j]) * deltas.get(1).get(level)[i][j]);
                    neuralNet.get(1).get(level)[i][j] += variationsPrevious.get(1).get(level)[i][j];
                    gradientsPrevious.get(1).get(level)[i][j] = gradients.get(1).get(level)[i][j];

                } else if ((gradientsPrevious.get(1).get(level)[i][j] * gradients.get(1).get(level)[i][j]) < 0) {
                    deltas.get(1).get(level)[i][j] = Math.max(deltas.get(1).get(level)[i][j] * decreaseFactor, minDelta);

                    //Comment the IF conditional below to use the Rprop+
                    //If not commented, the algorithm used is the iRprop+
                    //iRprop+ was proposed by Christian Igel and Michael Husken - Improving the Rprop Learning Algorithm, 2000
                    //if (currentError > previousError) {
                    neuralNet.get(1).get(level)[i][j] -= variationsPrevious.get(1).get(level)[i][j];
                    //}
                    //gradientsPrevious.get(1).get(level)[i][j] = gradients.get(1).get(level)[i][j];
                    gradientsPrevious.get(1).get(level)[i][j] = 0.0;

                } else if ((gradientsPrevious.get(1).get(level)[i][j] * gradients.get(1).get(level)[i][j]) == 0) {
                    variationsPrevious.get(1).get(level)[i][j] = -(Math.signum(gradients.get(1).get(level)[i][j]) * deltas.get(1).get(level)[i][j]);
                    neuralNet.get(1).get(level)[i][j] += variationsPrevious.get(1).get(level)[i][j];
                    gradientsPrevious.get(1).get(level)[i][j] = gradients.get(1).get(level)[i][j];
                }


                //System.out.println("Gradients = " + gradients.get(1).get(level)[0][0]);
                //System.out.println("Deltas = " + deltas.get(1).get(level)[0][0]);
            }
        }
    }

    /*===========================================================================
     * Update the weights between an input layer and an hidden layer - BP
     *===========================================================================*/
    static void updateWeightsInputLayerHiddenLayer(ArrayList<Integer[]> arrayArchitecture,
            ArrayList<Double> localGradientHidden,
            ArrayList<ArrayList<Double[]>> outputs,
            ArrayList<ArrayList<Double[][]>> neuralNet,
            ArrayList<ArrayList<Double[][]>> variationsPrevious,
            double learningRate, double momentumConstant, int level, int numInInst,
            ArrayList<ArrayList<ArrayList<Double>>> biasWeights,
            ArrayList<ArrayList<ArrayList<Double>>> biasVariationsPrevious,
            ArrayList<ArrayList<Double>> datasetTrain,
            double weightDecay,
            ArrayList<ArrayList<Integer>> indexesLevels) {

        for (int j = 0; j < arrayArchitecture.get(1)[level]; j++) {
            double delta = 0;
            double deltaBias = learningRate * localGradientHidden.get(j) * 1;

            //Update bias weights
            if (numInInst == 0) {
                double bias = biasWeights.get(0).get(level).get(j) + deltaBias;
                biasWeights.get(0).get(level).set(j, bias);
                biasVariationsPrevious.get(0).get(level).set(j, deltaBias);
            } else {
                double bias = biasWeights.get(0).get(level).get(j) + momentumConstant * biasVariationsPrevious.get(0).get(level).get(j) + deltaBias;
                biasWeights.get(0).get(level).set(j, bias);
                biasVariationsPrevious.get(0).get(level).set(j, deltaBias);
            }

            //Update neural net weights
            for (int i = 0; i < arrayArchitecture.get(0)[level]; i++) {//Aqui utiliza exemplos+classes
            //for (int i = 0; i < arrayArchitecture.get(0)[0]; i++) {//Aqui utiliza so exemplos

                double weightDecayContrib = learningRate * weightDecay * (neuralNet.get(0).get(level)[i][j] / (Math.pow(1 + Math.pow(neuralNet.get(0).get(level)[i][j], 2), 2)));
                //IF-ELSE descomentados sao para usar exemplos+classes
                if (i < arrayArchitecture.get(0)[0]) {
                    delta = learningRate * localGradientHidden.get(j) * datasetTrain.get(numInInst).get(i) - weightDecayContrib;
                } else {
                    //Predicted labels as feature vectors
                    delta = learningRate * localGradientHidden.get(j) * outputs.get(1).get(level - 1)[i - arrayArchitecture.get(0)[0]] - weightDecayContrib;
                    
                    //True labels as feature vectors
                    //int indexClass = indexesLevels.get(level-1).get(i - arrayArchitecture.get(0)[0]);
                    //delta = learningRate * localGradientHidden.get(j) * datasetTrain.get(numInInst).get(indexClass) - weightDecayContrib;
                }

                if (numInInst == 0) {
                    neuralNet.get(0).get(level)[i][j] = neuralNet.get(0).get(level)[i][j] + delta;
                    variationsPrevious.get(0).get(level)[i][j] = delta;
                } else {
                    neuralNet.get(0).get(level)[i][j] = neuralNet.get(0).get(level)[i][j] + momentumConstant * variationsPrevious.get(0).get(level)[i][j] + delta;
                    variationsPrevious.get(0).get(level)[i][j] = delta;
                }
            }
        }
    }

    /*===========================================================================
     * Update the weights between an input layer and an hidden layer - Rprop
     *===========================================================================*/
    static void updateWeightsInputLayerHiddenLayer(ArrayList<Integer[]> arrayArchitecture,
            ArrayList<ArrayList<Double[][]>> gradients,
            ArrayList<ArrayList<ArrayList<Double>>> biasWeights,
            ArrayList<ArrayList<ArrayList<Double>>> biasVariationsPrevious,
            ArrayList<ArrayList<Double[][]>> deltas,
            ArrayList<ArrayList<Double[][]>> deltasPrevious,
            ArrayList<ArrayList<Double[][]>> gradientsPrevious,
            ArrayList<ArrayList<Double[][]>> neuralNet,
            ArrayList<ArrayList<Double[][]>> variationsPrevious,
            ArrayList<ArrayList<ArrayList<Double>>> deltasBias,
            ArrayList<ArrayList<ArrayList<Double>>> deltasBiasPrevious,
            ArrayList<ArrayList<ArrayList<Double>>> gradientsBias,
            ArrayList<ArrayList<ArrayList<Double>>> gradientsBiasPrevious,
            double increaseFactor,
            double decreaseFactor,
            double initialDelta,
            double maxDelta,
            double minDelta,
            int level,
            double currentError,
            double previousError) {

        for (int j = 0; j < arrayArchitecture.get(1)[level]; j++) {

            if ((gradientsBiasPrevious.get(0).get(level).get(j) * gradientsBias.get(0).get(level).get(j)) > 0) {
                deltasBias.get(0).get(level).set(j, Math.min(deltasBias.get(0).get(level).get(j) * increaseFactor, maxDelta));
                biasVariationsPrevious.get(0).get(level).set(j, -(Math.signum(gradientsBias.get(0).get(level).get(j)) * deltasBias.get(0).get(level).get(j)));
                biasWeights.get(0).get(level).set(j, biasWeights.get(0).get(level).get(j) + biasVariationsPrevious.get(0).get(level).get(j));
                gradientsBiasPrevious.get(0).get(level).set(j, gradientsBias.get(0).get(level).get(j));

            } else if ((gradientsBiasPrevious.get(0).get(level).get(j) * gradientsBias.get(0).get(level).get(j)) < 0) {
                deltasBias.get(0).get(level).set(j, Math.max(deltasBias.get(0).get(level).get(j) * decreaseFactor, minDelta));

                //Comment the IF conditional below to use the Rprop+
                //If not commented, the algorithm used is the iRprop+
                //iRprop+ was proposed by Christian Igel and Michael Husken - Improving the Rprop Learning Algorithm, 2000
                //if (currentError > previousError) {
                biasWeights.get(0).get(level).set(j, biasWeights.get(0).get(level).get(j) - biasVariationsPrevious.get(0).get(level).get(j));
                //}
                //gradientsBiasPrevious.get(0).get(level).set(j, gradientsBias.get(0).get(level).get(j));
                gradientsBiasPrevious.get(0).get(level).set(j, 0.0);

            } else if ((gradientsBiasPrevious.get(0).get(level).get(j) * gradientsBias.get(0).get(level).get(j)) == 0) {
                biasVariationsPrevious.get(0).get(level).set(j, -(Math.signum(gradientsBias.get(0).get(level).get(j)) * deltasBias.get(0).get(level).get(j)));
                biasWeights.get(0).get(level).set(j, biasWeights.get(0).get(level).get(j) + biasVariationsPrevious.get(0).get(level).get(j));
                gradientsBiasPrevious.get(0).get(level).set(j, gradientsBias.get(0).get(level).get(j));
            }

            //Update neural net weights
            for (int i = 0; i < arrayArchitecture.get(0)[level]; i++) { //Aqui utiliza exemplos+classes
            //for (int i = 0; i < arrayArchitecture.get(0)[0]; i++) {//Aqui utiliza so exemplos

                if ((gradientsPrevious.get(0).get(level)[i][j] * gradients.get(0).get(level)[i][j]) > 0) {
                    deltas.get(0).get(level)[i][j] = Math.min(deltas.get(0).get(level)[i][j] * increaseFactor, maxDelta);
                    variationsPrevious.get(0).get(level)[i][j] = -(Math.signum(gradients.get(0).get(level)[i][j]) * deltas.get(0).get(level)[i][j]);
                    neuralNet.get(0).get(level)[i][j] += variationsPrevious.get(0).get(level)[i][j];
                    gradientsPrevious.get(0).get(level)[i][j] = gradients.get(0).get(level)[i][j];

                } else if ((gradientsPrevious.get(0).get(level)[i][j] * gradients.get(0).get(level)[i][j]) < 0) {
                    deltas.get(0).get(level)[i][j] = Math.max(deltas.get(0).get(level)[i][j] * decreaseFactor, minDelta);

                    //Comment the IF conditional below to use the Rprop+
                    //If not commented, the algorithm used is the iRprop+
                    //iRprop+ was proposed by Christian Igel and Michael Husken - Improving the Rprop Learning Algorithm, 2000
                    //if (currentError > previousError) {
                    neuralNet.get(0).get(level)[i][j] -= variationsPrevious.get(0).get(level)[i][j];
                    //}
                    //gradientsPrevious.get(0).get(level)[i][j] = gradients.get(0).get(level)[i][j];
                    gradientsPrevious.get(0).get(level)[i][j] = 0.0;

                } else if ((gradientsPrevious.get(0).get(level)[i][j] * gradients.get(0).get(level)[i][j]) == 0) {
                    variationsPrevious.get(0).get(level)[i][j] = -(Math.signum(gradients.get(0).get(level)[i][j]) * deltas.get(0).get(level)[i][j]);
                    neuralNet.get(0).get(level)[i][j] += variationsPrevious.get(0).get(level)[i][j];
                    gradientsPrevious.get(0).get(level)[i][j] = gradients.get(0).get(level)[i][j];
                }
            }
        }
    }

    /*===========================================================================
     * Verify and correct possible inconsistencies in the predictions
     *===========================================================================*/
    static int[][] verifyInconsistencies2(int[][] matrixOutputs,
            ArrayList<ArrayList<Integer>> indexesLevels,
            ArrayList<String> namesAttributes) {

        int indexFirstClass = indexesLevels.get(0).get(0);

        for (int numInst = 0; numInst < matrixOutputs.length; numInst++) {
            for (int i = indexesLevels.size() - 1; i > 0; i--) {
                //for(int i=indexesLevels.size()-1; i>=0; i--){
                for (int j = 0; j < indexesLevels.get(i).size(); j++) {
                    int index = indexesLevels.get(i).get(j);
                    if (matrixOutputs[numInst][index] == 1) {
                        String nameClass = namesAttributes.get(index);
                        String[] allClasses = nameClass.split("\\.");
                        String regExp = "^";
                        int pos = 0;
                        regExp = regExp + allClasses[pos];
                        Pattern pattern = Pattern.compile(regExp + "$");
                        for (int k = indexFirstClass; k < namesAttributes.size(); k++) {
                            Matcher m = pattern.matcher(namesAttributes.get(k));
                            if (m.find()) {
                                if (pos < allClasses.length - 1) {
                                    matrixOutputs[numInst][k] = 1;
                                    pos++;
                                    regExp = regExp + "\\.";
                                    regExp = regExp + allClasses[pos];
                                    pattern = Pattern.compile(regExp + "$");
                                } else {
                                    break;
                                }
                            }
                        }
                    }
                }
            }
        }
        return matrixOutputs;
    }

    static int[][] verifyInconsistencies(int[][] matrixOutputs,
            ArrayList<ArrayList<Integer>> indexesLevels,
            ArrayList<String> namesAttributes,
            String hierarchyType) {

        int indexFirstClass = indexesLevels.get(0).get(0);

        if (hierarchyType.equals("DAG")) {

            ArrayList<ArrayList<String[]>> allDAGClasses = Classes.getAllPathsDAGClasses();

            for (int numInst = 0; numInst < matrixOutputs.length; numInst++) {
                for (int i = indexesLevels.size() - 1; i > 0; i--) {
                    for (int j = 0; j < indexesLevels.get(i).size(); j++) {

                        int index = indexesLevels.get(i).get(j);

                        if (matrixOutputs[numInst][index] == 1) {//The class was predicted

                            //Search for the index in allDAGClasses
                            //The indexes in R start with 1, and the classes in matrixOutputs
                            //start in position numAttributes+1. so lets search for (index-indexesLevels.get(0).get(0))+1
                            int indexToSearch = (index - indexesLevels.get(0).get(0)) + 1;
                            int indexClass = getIndexAllDAGClasses(indexToSearch, allDAGClasses);

                            //Verify if at least one path of the DAG was predicted
                            //Store the positions where a 0 was found in the paths between the root node and he class
                            ArrayList<Integer> positions = new ArrayList<Integer>();
                            int path = verifyIfDAGPathPredicted(indexClass, allDAGClasses, matrixOutputs[numInst], indexesLevels.get(0).get(0), positions);

                            if (path == 1) {//There was at least one path predicted, so set all the superclasses to 1
                                correctPrediction(numInst, indexClass, allDAGClasses, matrixOutputs, indexesLevels.get(0).get(0));
                            } else {//No complete path predicted, so set matrixOutputs[numInst][index] to 0
                                //matrixOutputs[numInst][index] = 0;
                                //In all paths, set all positions from where 0 was found to matrixOutputs[numInst][index]
                                correctPrediction(numInst, indexClass, allDAGClasses, matrixOutputs, indexesLevels.get(0).get(0), positions);
                            }
                        }
                    }
                }
            }

        } else {
            for (int numInst = 0; numInst < matrixOutputs.length; numInst++) {
                for (int i = indexesLevels.size() - 1; i > 0; i--) {
                    for (int j = 0; j < indexesLevels.get(i).size(); j++) {
                        int index = indexesLevels.get(i).get(j);
                        if (matrixOutputs[numInst][index] == 1) {
                            String nameClass = namesAttributes.get(index);
                            String[] allClasses = nameClass.split("\\.");

                            ArrayList<Integer> vectPos = new ArrayList<Integer>();
                            ArrayList<Integer> vectVal = new ArrayList<Integer>();
                            vectPos.add(index);
                            vectVal.add(1);

                            String regExp = "^";
                            int pos = 0;
                            regExp = regExp + allClasses[pos];
                            Pattern pattern = Pattern.compile(regExp + "$");

                            for (int k = indexFirstClass; k < namesAttributes.size(); k++) {
                                Matcher m = pattern.matcher(namesAttributes.get(k));
                                if (m.find()) {
                                    if (pos < allClasses.length - 1) {

                                        vectPos.add(vectPos.size() - 1, k);
                                        vectVal.add(vectVal.size() - 1, matrixOutputs[numInst][k]);

                                        pos++;
                                        regExp = regExp + "\\.";
                                        regExp = regExp + allClasses[pos];
                                        pattern = Pattern.compile(regExp + "$");
                                    } else {
                                        break;
                                    }
                                }
                            }

                            correctPrediction(vectVal, vectPos, matrixOutputs, numInst);

                        }
                    }
                }
            }
        }

        return matrixOutputs;
    }

    /*===========================================================================
     * Correct a given prediction from inconsistencies (Tree)
     *===========================================================================*/
    static void correctPrediction(ArrayList<Integer> vectVal, ArrayList<Integer> vectPos,
            int[][] matrixOutputs, int numInst) {

        for (int i = (vectVal.size() - 1); i > 0; i--) {
            if (vectVal.get(i) == 1 && vectVal.get(i - 1) == 0) {

                for (int j = i; j < vectPos.size(); j++) {
                    int p = vectPos.get(j);
                    matrixOutputs[numInst][p] = 0;
                }
            }
        }
    }

    /*===========================================================================
     * Correct a given prediction from inconsistencies (DAG)
     *===========================================================================*/
    static void correctPrediction(int numInst, int indexClass, ArrayList<ArrayList<String[]>> allDAGClasses,
            int[][] matrixOutputs, int numAttributes) {

        for (int i = 1; i < allDAGClasses.get(indexClass).size(); i++) {
            for (int j = 0; j < allDAGClasses.get(indexClass).get(i).length; j++) {
                int indexToSet = Integer.parseInt(allDAGClasses.get(indexClass).get(i)[j]);
                indexToSet = (indexToSet + numAttributes) - 1;
                matrixOutputs[numInst][indexToSet] = 1;
            }
        }
    }
    
    static void correctPrediction(int numInst, int indexClass, ArrayList<ArrayList<String[]>> allDAGClasses,
            int[][] matrixOutputs, int numAttributes, ArrayList<Integer> positions) {

        for (int i = 1; i < allDAGClasses.get(indexClass).size(); i++) {
            int posZero = positions.get(i-1);
            for (int j = posZero; j < allDAGClasses.get(indexClass).get(i).length; j++) {
                int indexToSet = Integer.parseInt(allDAGClasses.get(indexClass).get(i)[j]);
                indexToSet = (indexToSet + numAttributes) - 1;
                matrixOutputs[numInst][indexToSet] = 0;
            }
        }
    }

    /*===========================================================================
     * Search for an index class in allDAGClasses
     *===========================================================================*/
    static int getIndexAllDAGClasses(int index, ArrayList<ArrayList<String[]>> allDAGClasses) {

        int indexClass = 0;

        for (int i = 0; i < allDAGClasses.size(); i++) {
            if (Integer.parseInt(allDAGClasses.get(i).get(0)[0]) == index) {
                indexClass = i;
                break;
            }
        }

        return indexClass;
    }

    /*===========================================================================
     * Given the index of a class, verify if at least one of its path to the root
     * was completely predicted
     *===========================================================================*/
    /*static int verifyIfDAGPathPredicted(int indexClass,
            ArrayList<ArrayList<String[]>> allDAGClasses,
            int[] outputInstance,
            int numAttributes) {

        int predicted = 0;

        for (int i = 1; i < allDAGClasses.get(indexClass).size(); i++) {
            predicted = 1;
            for (int j = 0; j < allDAGClasses.get(indexClass).get(i).length; j++) {
                int indexToVerify = Integer.parseInt(allDAGClasses.get(indexClass).get(i)[j]);
                indexToVerify = (indexToVerify + numAttributes) - 1;
                if (outputInstance[indexToVerify] == 0) {
                    predicted = 0;
                    break;
                }
            }
            if (predicted == 1) {
                break;
            }
        }

        return predicted;
    }*/
    static int verifyIfDAGPathPredicted(int indexClass,
            ArrayList<ArrayList<String[]>> allDAGClasses,
            int[] outputInstance,
            int numAttributes,
            ArrayList<Integer> positions) {

        int predicted = 0;

        for (int i = 1; i < allDAGClasses.get(indexClass).size(); i++) {
            predicted = 1;
            for (int j = 0; j < allDAGClasses.get(indexClass).get(i).length; j++) {
                int indexToVerify = Integer.parseInt(allDAGClasses.get(indexClass).get(i)[j]);
                indexToVerify = (indexToVerify + numAttributes) - 1;
                if (outputInstance[indexToVerify] == 0) {
                    predicted = 0;
                    positions.add(j);
                    break;
                }
            }
            if (predicted == 1) {
                break;
            }
        }

        return predicted;
    }

    /*===========================================================================
     * Retrieve the predictions
     *===========================================================================*/
    static String retrievePrediction(int[] vectorOutputs,
            ArrayList<String> namesAttributes,
            ArrayList<ArrayList<Integer>> indexesLevels) {

        ArrayList<String> predictedClasses = new ArrayList<String>();

        for (int i = (indexesLevels.size() - 1); i >= 0; i--) {
            for (int j = 0; j < indexesLevels.get(i).size(); j++) {
                int index = indexesLevels.get(i).get(j);
                if (vectorOutputs[index] == 1) {
                    if (i == (indexesLevels.size() - 1)) {
                        predictedClasses.add(namesAttributes.get(index));
                    } else {
                        int index2 = predictedClasses.size();
                        String nameClass = namesAttributes.get(index);
                        nameClass = "^" + nameClass + "\\.";
                        Pattern pattern = Pattern.compile(nameClass);
                        int found = 0;
                        for (int k = 0; k < index2; k++) {
                            Matcher m = pattern.matcher(predictedClasses.get(k));
                            if (m.find()) {
                                found = 1;
                                break;
                            }
                        }
                        if (found == 0) {
                            predictedClasses.add(namesAttributes.get(index));
                        }
                    }
                }
            }
        }

        String predClasses = "";
        for (int i = 0; i < predictedClasses.size(); i++) {
            if (i == 0) {
                predClasses = predictedClasses.get(i);
            } else {
                predClasses = predClasses + "@" + predictedClasses.get(i);
            }
        }
        return predClasses;
    }

    /*===========================================================================
     * Initialize the deltas used in the Rprop neural network
     *===========================================================================*/
    static ArrayList<ArrayList<Double[][]>> initializeDeltas(ArrayList<Integer[]> arrayArchitecture, double initialDelta) {

        ArrayList<ArrayList<Double[][]>> initialDeltas = new ArrayList<ArrayList<Double[][]>>();

        ArrayList<Double[][]> deltas1 = new ArrayList<Double[][]>();
        ArrayList<Double[][]> deltas2 = new ArrayList<Double[][]>();

        for (int i = 0; i < (arrayArchitecture.get(0).length - 1); i++) {
            Double matrixDeltas[][];

            //Deltas between an input layer and a hidden layer
            int numRows = arrayArchitecture.get(0)[i];//Aqui utiliza exemplos+classes
            //int numRows = arrayArchitecture.get(0)[0];//Aqui utiliza so exemplos
            int numColumns = arrayArchitecture.get(1)[i];
            matrixDeltas = new Double[numRows][numColumns];

            for (int j = 0; j < numRows; j++) {
                for (int k = 0; k < numColumns; k++) {
                    matrixDeltas[j][k] = initialDelta;
                }
            }
            deltas1.add(matrixDeltas);

            //Deltas between a hidden layer and an output layer
            numRows = arrayArchitecture.get(1)[i];
            numColumns = arrayArchitecture.get(0)[i + 1] - arrayArchitecture.get(0)[0]; //Aqui utiliza exemplos+classes
            //numColumns = arrayArchitecture.get(0)[i + 1]; //Aqui utiliza so exemplos
            matrixDeltas = new Double[numRows][numColumns];

            for (int j = 0; j < numRows; j++) {
                for (int k = 0; k < numColumns; k++) {
                    matrixDeltas[j][k] = initialDelta;
                }
            }
            deltas2.add(matrixDeltas);
        }

        initialDeltas.add(deltas1);
        initialDeltas.add(deltas2);

        return initialDeltas;
    }

    /*===========================================================================
     * Get the initial deltas of the biases
     *===========================================================================*/
    static ArrayList<ArrayList<ArrayList<Double>>> initializeBiasDeltasGradients(ArrayList<Integer[]> arrayArchitecture, double initialDelta) {

        ArrayList<ArrayList<ArrayList<Double>>> biasDeltas = new ArrayList<ArrayList<ArrayList<Double>>>();
        ArrayList<ArrayList<Double>> biasDeltas1 = new ArrayList<ArrayList<Double>>();
        ArrayList<ArrayList<Double>> biasDeltas2 = new ArrayList<ArrayList<Double>>();

        for (int i = 0; i < (arrayArchitecture.get(1).length - 1); i++) {

            int numNeurons = arrayArchitecture.get(1)[i];
            ArrayList<Double> deltas = new ArrayList<Double>();

            for (int j = 0; j < numNeurons; j++) {
                deltas.add(initialDelta);
            }

            biasDeltas1.add(deltas);
        }

        for (int i = 1; i < (arrayArchitecture.get(0).length); i++) {

            int numNeurons = arrayArchitecture.get(0)[i] - arrayArchitecture.get(0)[0]; //Aqui utiliza exemplos+classes
            //int numNeurons = arrayArchitecture.get(0)[i]; //Aqui utiliza so exemplos
            ArrayList<Double> deltas = new ArrayList<Double>();

            for (int j = 0; j < numNeurons; j++) {
                deltas.add(initialDelta);
            }

            biasDeltas2.add(deltas);
        }

        biasDeltas.add(biasDeltas1);
        biasDeltas.add(biasDeltas2);

        return biasDeltas;
    }

    /*===========================================================================
     * Initialize the gradients array used in the Rprop neural network
     *===========================================================================*/
    static ArrayList<ArrayList<Double[][]>> initializeGradients(ArrayList<Integer[]> arrayArchitecture) {

        ArrayList<ArrayList<Double[][]>> gradients = new ArrayList<ArrayList<Double[][]>>();

        ArrayList<Double[][]> grads1 = new ArrayList<Double[][]>();
        ArrayList<Double[][]> grads2 = new ArrayList<Double[][]>();

        for (int i = 0; i < (arrayArchitecture.get(0).length - 1); i++) {
            Double matrixGradients[][];

            //Gradients between an input layer and a hidden layer
            int numRows = arrayArchitecture.get(0)[i];//Aqui utiliza exemplos+classes
            //int numRows = arrayArchitecture.get(0)[0];//Aqui utiliza so exemplos
            int numColumns = arrayArchitecture.get(1)[i];
            matrixGradients = new Double[numRows][numColumns];

            for (int j = 0; j < numRows; j++) {
                for (int k = 0; k < numColumns; k++) {
                    matrixGradients[j][k] = 0.0;
                }
            }
            grads1.add(matrixGradients);

            //Gradients between a hidden layer and an output layer
            numRows = arrayArchitecture.get(1)[i];
            numColumns = arrayArchitecture.get(0)[i + 1] - arrayArchitecture.get(0)[0];//Aqui utiliza exemplos+classes
            //numColumns = arrayArchitecture.get(0)[i + 1];//Aqui utiliza so exemplos
            matrixGradients = new Double[numRows][numColumns];

            for (int j = 0; j < numRows; j++) {
                for (int k = 0; k < numColumns; k++) {
                    matrixGradients[j][k] = 0.0;
                }
            }
            grads2.add(matrixGradients);
        }

        gradients.add(grads1);
        gradients.add(grads2);

        return gradients;
    }

    /*===========================================================================
     * Calculate gradients between the first hidden layer and the first level layer
     *===========================================================================*/
    static void calculateGradientsFHLayerFLLayer(ArrayList<Integer[]> arrayArchitecture,
            ArrayList<ArrayList<Double[]>> outputs,
            ArrayList<Double> localGradientOutput,
            ArrayList<ArrayList<Double[][]>> gradients,
            ArrayList<ArrayList<ArrayList<Double>>> gradientsBias,
            int numInInst) {

        for (int j = 0; j < arrayArchitecture.get(0)[1] - arrayArchitecture.get(0)[0]; j++) {//Aqui utiliza exemplos+classes
        //for (int j = 0; j < arrayArchitecture.get(0)[1]; j++) {//Aqui utiliza so exemplos

            double param = -localGradientOutput.get(j) * 1;
            double bias = gradientsBias.get(1).get(0).get(j) + param;
            gradientsBias.get(1).get(0).set(j, bias);

            for (int i = 0; i < outputs.get(0).get(0).length; i++) {

                gradients.get(1).get(0)[i][j] += (-localGradientOutput.get(j) * outputs.get(0).get(0)[i]);

            }
        }
    }

    /*===========================================================================
     * Calculate the gradients between the input layer and the first hidden layer
     *===========================================================================*/
    static void calculateGradientsFILayerFHLayer(ArrayList<Integer[]> arrayArchitecture,
            ArrayList<Double> localGradientHidden,
            ArrayList<ArrayList<Double>> datasetTrain,
            ArrayList<ArrayList<Double[][]>> gradients,
            ArrayList<ArrayList<ArrayList<Double>>> gradientsBias,
            int numInInst) {

        for (int j = 0; j < arrayArchitecture.get(1)[0]; j++) {

            double param = -localGradientHidden.get(j) * 1;
            double bias = gradientsBias.get(0).get(0).get(j) + param;
            gradientsBias.get(0).get(0).set(j, bias);

            for (int i = 0; i < arrayArchitecture.get(0)[0]; i++) {

                gradients.get(0).get(0)[i][j] += (-localGradientHidden.get(j) * datasetTrain.get(numInInst).get(i));

            }
        }
    }

    /*===========================================================================
     * Calculate gradients between an hidden layer and a level layer
     *===========================================================================*/
    static void calculateGradientHiddenLayerLevelLayer(ArrayList<Integer[]> arrayArchitecture,
            ArrayList<ArrayList<Double[]>> outputs,
            ArrayList<Double> localGradientOutput,
            ArrayList<ArrayList<Double[][]>> gradients,
            ArrayList<ArrayList<ArrayList<Double>>> gradientsBias,
            int level) {

        for (int j = 0; j < arrayArchitecture.get(0)[level + 1] - arrayArchitecture.get(0)[0]; j++) {//Aqui utiliza exemplos+classes
        //for (int j = 0; j < arrayArchitecture.get(0)[level + 1]; j++) {//Aqui utiliza so exemplos

            double param = -localGradientOutput.get(j) * 1;
            double bias = gradientsBias.get(1).get(level).get(j) + param;
            gradientsBias.get(1).get(level).set(j, bias);

            for (int i = 0; i < outputs.get(0).get(level).length; i++) {

                gradients.get(1).get(level)[i][j] += (-localGradientOutput.get(j) * outputs.get(0).get(level)[i]);

            }
        }
    }

    /*===========================================================================
     * Calculate the gradients between an input layer and an hidden layer
     *===========================================================================*/
    static void calculateGradientInputLayerHiddenLayer(ArrayList<Integer[]> arrayArchitecture,
            ArrayList<Double> localGradientHidden,
            ArrayList<ArrayList<Double[]>> outputs,
            ArrayList<ArrayList<Double[][]>> gradients,
            ArrayList<ArrayList<ArrayList<Double>>> gradientsBias,
            ArrayList<ArrayList<Double>> datasetTrain,
            int numInInst,
            int level,
            ArrayList<ArrayList<Integer>> indexesLevels) {

        for (int j = 0; j < arrayArchitecture.get(1)[level]; j++) {

            double param = -localGradientHidden.get(j) * 1;
            double bias = gradientsBias.get(0).get(level).get(j) + param;
            gradientsBias.get(0).get(level).set(j, bias);

            for (int i = 0; i < arrayArchitecture.get(0)[level]; i++) {//Aqui utiliza exemplos+classes
            //for (int i = 0; i < arrayArchitecture.get(0)[0]; i++) {//Aqui utiliza so exemplos    
                //IF-ELSE descomentados sao para usar exemplos+classes
                if (i < arrayArchitecture.get(0)[0]) {
                    gradients.get(0).get(level)[i][j] += -localGradientHidden.get(j) * datasetTrain.get(numInInst).get(i);
                } else {
                    //Predicted labels as features
                    gradients.get(0).get(level)[i][j] += (-localGradientHidden.get(j) * outputs.get(1).get(level - 1)[i - arrayArchitecture.get(0)[0]]);
                    
                    //True labels as features
                    //int indexClass = indexesLevels.get(level-1).get(i - arrayArchitecture.get(0)[0]);
                    //gradients.get(0).get(level)[i][j] += (-localGradientHidden.get(j) * datasetTrain.get(numInInst).get(indexClass));
                }
            }
        }
    }

    /*===========================================================================
     * Create an ArrayList with the architecture of the neural network
     *===========================================================================*/
    static ArrayList<Integer[]> getNetArchitecture(int numLevels, ArrayList<Double> percentageUnitsHidden,
            ArrayList<ArrayList<Integer>> indexesLevels) {

        ArrayList<Integer[]> arrayListArchitecture = new ArrayList<Integer[]>();

        Integer vectorArchitecture1[];
        vectorArchitecture1 = new Integer[numLevels + 1];
        Integer vectorArchitecture2[];
        vectorArchitecture2 = new Integer[numLevels + 1];
        vectorArchitecture1[0] = indexesLevels.get(0).get(0);

        vectorArchitecture2[0] = (int) Math.round(vectorArchitecture1[0] * percentageUnitsHidden.get(0));

        for (int i = 0; i < numLevels; i++) {

            vectorArchitecture1[i + 1] = vectorArchitecture1[0] + indexesLevels.get(i).size(); //Aqui utiliza exemplos+classes
            vectorArchitecture2[i + 1] = (int) Math.round(vectorArchitecture1[i + 1] * percentageUnitsHidden.get(i + 1));//Aqui utiliza exemplos+classes
            
            //vectorArchitecture1[i + 1] = indexesLevels.get(i).size(); //Aqui utiliza so exemplos
            //vectorArchitecture2[i + 1] = (int) Math.round(vectorArchitecture1[0] * percentageUnitsHidden.get(i + 1));//Aqui utiliza so exemplos
        }

        arrayListArchitecture.add(vectorArchitecture1);
        arrayListArchitecture.add(vectorArchitecture2);

        return (arrayListArchitecture);
    }

    /*===========================================================================
     * Gets the position of a specific class in the atributes
     * String regExpClass must in regular expression format
     * Ex: 2\\.1\\.10
     *===========================================================================*/
    static int getPosSpecificClass(String regExpClass, ArrayList<String> namesAttributes,
            int indexFirstClass) {

        int pos = 0;

        Pattern pattern = Pattern.compile("^" + regExpClass + "$");

        for (int i = indexFirstClass; i < namesAttributes.size(); i++) {
            Matcher m = pattern.matcher(namesAttributes.get(i));
            if (m.find()) {
                pos = i;
                break;
            }
        }

        return pos;
    }

    /*===========================================================================
     * Puts a vector of classes in a regular expression format
     * Ex: [2,1,10] --> 2\\.1\\.10
     *===========================================================================*/
    static String mountExpRegFromVectorClasses(String[] vectorClasses, int posLastClass) {

        String regExpClass = "";

        for (int i = 0; i < posLastClass; i++) {
            if (i < (posLastClass - 1)) {
                regExpClass = regExpClass + vectorClasses[i] + "\\.";
            } else {
                regExpClass = regExpClass + vectorClasses[i];
            }
        }

        return regExpClass;

    }

    /*===========================================================================
     * Puts a class in a regular expression format
     * Ex: 2.1.10 --> 2\\.1\\.10
     *===========================================================================*/
    static String putClassRegExpFormat(String classString) {

        String regExpClass = "";

        String[] allClasses = classString.split("\\.");

        for (int i = 0; i < allClasses.length; i++) {
            if (i < (allClasses.length - 1)) {
                regExpClass = regExpClass + allClasses[i] + "\\.";
            } else {
                regExpClass = regExpClass + allClasses[i];
            }
        }

        return regExpClass;
    }

    /*===========================================================================
     * Verify if a given class is a leaf node of the hierarchy
     *===========================================================================*/
    static int verifyIsLeaf(String stringClass, ArrayList<String> namesAttributes,
            int indexFirstClass) {

        int isLeaf = 1;

        String regExpClass = putClassRegExpFormat(stringClass);

        Pattern pattern = Pattern.compile("^" + regExpClass + "\\.");

        for (int i = indexFirstClass; i < namesAttributes.size(); i++) {
            Matcher m = pattern.matcher(namesAttributes.get(i));
            if (m.find()) {
                isLeaf = 0;
                break;
            }
        }

        return isLeaf;
    }

    /*===========================================================================
     * Verify if there is a class predicted in the current level
     *===========================================================================*/
    static int verifyIsPredicted(ArrayList<Double> instance,
            int indexFirstClass, int indexLastClass) {

        int isPredicted = 0;

        for (int i = indexFirstClass; i <= indexLastClass; i++) {
            if (instance.get(i) == 1.0) {
                isPredicted = 1;
                break;
            }
        }

        return isPredicted;
    }

    /*===========================================================================
     * Calculate mean and standard deviation of a vector of results
     *===========================================================================*/
    static double[] calculateMeanSd(double[] vector) {

        double[] meanSd = new double[3];
        double sum = 0;
        double mean = 0;
        double sd = 0;

        for (int i = 0; i < vector.length; i++) {
            sum += vector[i];
        }
        mean = sum / (vector.length);
        meanSd[0] = mean;
        sum = 0;

        for (int i = 0; i < vector.length; i++) {
            sum += Math.pow((vector[i] - meanSd[0]), 2);
        }
        sum = sum / (vector.length - 1);
        sd = Math.sqrt(sum);

        meanSd[1] = sd;

        return meanSd;
    }

    static double[] calculateMeanSd(int[] vector) {

        double[] meanSd = new double[2];
        double sum = 0;
        double mean = 0;
        double sd = 0;

        for (int i = 0; i < vector.length; i++) {
            sum += vector[i];
        }
        mean = sum / (vector.length);
        meanSd[0] = mean;
        sum = 0;

        for (int i = 0; i < vector.length; i++) {
            sum += Math.pow((vector[i] - meanSd[0]), 2);
        }
        sum = sum / (vector.length - 1);
        sd = Math.sqrt(sum);

        meanSd[1] = sd;

        return meanSd;
    }

    /*===========================================================================
     * Calculate the means of the AUPRC of all executions for all classes 
     *===========================================================================*/
    static void calculateMeansAUPRCClasses(ArrayList<double[]> meansAllAUPRCClasses,
            ArrayList<ArrayList<double[]>> allAUPRCClasses) {

        //Store the values of all executions for one class
        double[] AUPRCClass = new double[allAUPRCClasses.size()];

        //Go through all classes
        for (int i = 0; i < allAUPRCClasses.get(0).size(); i++) {

            //Go through all executions
            for (int j = 0; j < allAUPRCClasses.size(); j++) {

                AUPRCClass[j] = allAUPRCClasses.get(j).get(i)[0];

            }

            //Calculate the mean and sd
            double[] meanSd = calculateMeanSd(AUPRCClass);
            meanSd[2] = allAUPRCClasses.get(0).get(i)[1];
            meansAllAUPRCClasses.add(meanSd);

        }
    }

    /*===========================================================================
     * Get the higher value of the neural net for one given level
     *===========================================================================*/
    static int getHigherValue(double[][] matrixOutpustD, int indexFirstClass,
            int indexLastClass, int numInst) {


        double maxValue = 0;
        int maxIndex = 0;

        for (int i = indexFirstClass; i <= indexLastClass; i++) {

            if (matrixOutpustD[numInst][i] > maxValue) {
                maxValue = matrixOutpustD[numInst][i];
                maxIndex = i;
            }

        }
        return maxIndex;
    }

    /*===========================================================================
     * Calculate f-measure from precision and recall
     *===========================================================================*/
    static double calculateFmeasure(double precision, double recall) {

        double fMeasure = 0;

        fMeasure = (2 * precision * recall) / (precision + recall);

        return fMeasure;
    }
}
