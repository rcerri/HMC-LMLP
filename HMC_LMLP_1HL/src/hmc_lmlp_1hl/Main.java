/* Hierarchical Multilabel Classification with Local Multi-Layer Perceptrons.
 * The conventional back-propagation and the resilient back-propagation are implemented.
 * The program implements the conventional error calculation, and also implements
 * the initial idea of the error function proposed by:
 *
 * Min-Ling Zhang and Zhi-Hua Zhou
 * Multilabel Neural Networks with Applications to Functional Genomics and Text Categorization
 *
 * The program is not object oriented, although written in Java.
 * That's bad, I know! What a pity! But it works anyway...
 *
 * Absolutely no guarantees or warranties are made concerning the suitability,
 * correctness, or any other aspect of this program. Any use is at your own risk.
 */
package hmc_lmlp_1hl;

import java.util.*;

/**
 * @author Ricardo Cerri
 */
public class Main {

    /*
     * @param args the command line arguments
     */
    public static void main(String[] args) {

        String configFile = args[0];

        double weightDecay = 0;
        double increaseFactor = 0;
        double decreaseFactor = 0;
        double initialDelta = 0;
        double maxDelta = 0;
        double minDelta = 0;
        ArrayList<ArrayList<Double>> momentumAndLearning = new ArrayList<ArrayList<Double>>();
        ArrayList<Double> learningRate = new ArrayList<Double>();
        ArrayList<Double> momentumConstant = new ArrayList<Double>();
        ArrayList<Double> rpropParam = new ArrayList<Double>();

        //Get the parameters in the configuration file
        int learningAlgorithm = FunctionsCommon.getLearningAlgorithm(configFile);
        ArrayList<Integer> numEpochs = FunctionsCommon.getNumEpochs(configFile);
        double thresholdReductionFactor = FunctionsMultiLabel.getThresholdReduction(configFile);

        //Dataset is multi-label or single-label?
        int multilabel = FunctionsCommon.getMultiLabel(configFile);

        //Parameters of the back-propagation algorithm
        if (learningAlgorithm == 1) {
            momentumAndLearning = FunctionsCommon.getMomentumAndLearning(configFile);
            learningRate = momentumAndLearning.get(0);
            momentumConstant = momentumAndLearning.get(1);
            weightDecay = FunctionsCommon.getWeightDecay(configFile);
            weightDecay = FunctionsCommon.verifyWeightDecay(weightDecay, learningRate.get(0), numEpochs.get(numEpochs.size() - 1));
        } //Parameters of the resilient back-propagation algorithm
        else {
            rpropParam = FunctionsCommon.getRpropParam(configFile);
            increaseFactor = rpropParam.get(0);
            decreaseFactor = rpropParam.get(1);
            initialDelta = rpropParam.get(2);
            maxDelta = rpropParam.get(3);
            minDelta = rpropParam.get(4);
            weightDecay = 0;
        }

        int numRuns = FunctionsCommon.getNumberRuns(configFile);
        int numLevels = FunctionsCommon.getNumLevels(configFile);
        ArrayList<Double> thresholdValues = FunctionsMultiLabel.getThresholdValues(configFile);
        ArrayList<Double> percentageUnitsHidden = FunctionsCommon.getPercentageHiddenUnits(configFile);
        int errChoice = FunctionsCommon.getErrorChoice(configFile);
        int printPredictions = FunctionsCommon.getPrintPredictions(configFile);

        //Store results for all runs to calculate mean and standard deviation
        double[] allAUPRCValid = new double[numRuns];
        double[] allAUPRCTest = new double[numRuns];
        int[] allEpochs = new int[numRuns];
        double[] allMsTimes = new double[numRuns];
        double[] allSTimes = new double[numRuns];
        double[] allMTimes = new double[numRuns];
        double[] allHTimes = new double[numRuns];

        ArrayList<ArrayList<double[]>> allAUPRCClasses = new ArrayList<ArrayList<double[]>>();
        ArrayList<double[]> meansAllAUPRCClasses = new ArrayList<double[]>();

        //Path of all datasets
        String pathDatasets = FunctionsCommon.getPathDatasets(configFile);

        String hierarchyType = FunctionsCommon.getHierarchyType(configFile);
        if (hierarchyType.equals("DAG")) {
            Classes.setClassesPaths(pathDatasets,configFile);
        }

        ArrayList<ArrayList<String>> namesDatasets = FunctionsCommon.getNamesData(configFile);

        //Names of the datasets
        ArrayList<String> nameDatasetTrain = namesDatasets.get(0);
        ArrayList<String> nameDatasetValid = namesDatasets.get(1);
        ArrayList<String> nameDatasetTest = namesDatasets.get(2);

        //Execute for all datasets
        //Generally, there are various datasets if one is doing cross-validation
        //But then there is just one run
        for (int numData = 0; numData < nameDatasetTrain.size(); numData++) {

            //Create directories for outputs
            System.out.println("Creating directories...");
            FunctionsCommon.createDirectories(nameDatasetTest.get(numData), numEpochs,
                    thresholdValues, errChoice, numRuns, learningAlgorithm);

            //Get the names of the classes
            ArrayList<String> namesAttributes = FunctionsCommon.getNamesAttributes(pathDatasets + nameDatasetTrain.get(numData));

            //Get the indexes of the classes in each hierarchical level of the training data
            ArrayList<ArrayList<Integer>> indexesLevels = FunctionsCommon.getIndexesLevels(pathDatasets + nameDatasetTrain.get(numData),
                    numLevels);

            //Train data
            ArrayList<ArrayList<Double>> datasetTrain = FunctionsCommon.readDataset(pathDatasets + nameDatasetTrain.get(numData));
            //Valid data
            ArrayList<ArrayList<Double>> datasetValid;
            if (nameDatasetValid.get(numData).equals("none")) {
                datasetValid = FunctionsCommon.getValidDataset(datasetTrain);
            } else {
                datasetValid = FunctionsCommon.readDataset(pathDatasets + nameDatasetValid.get(numData));
            }
            //Test data
            ArrayList<ArrayList<Double>> datasetTest = FunctionsCommon.readDataset(pathDatasets + nameDatasetTest.get(numData));

            //Net architecture
            ArrayList<Integer[]> arrayArchitecture = FunctionsCommon.getNetArchitecture(numLevels, percentageUnitsHidden, indexesLevels);

            //Obtain default classes
            double[] defaultClasses = FunctionsMultiLabel.obtainDefaultClasses(datasetTrain, indexesLevels, numLevels);

            //Training and testing process of the neural network
            if (learningAlgorithm == 1) {
                for (int nRun = 1; nRun <= numRuns; nRun++) {
                    HMC_LMLP_BP.hmcBpTrain(arrayArchitecture, numEpochs, datasetTrain, datasetValid, datasetTest, indexesLevels,
                            numLevels, learningRate, momentumConstant, thresholdValues, namesAttributes, nameDatasetValid.get(numData),
                            nameDatasetTest.get(numData), errChoice, nRun, weightDecay, printPredictions, learningAlgorithm,
                            thresholdReductionFactor, multilabel, allAUPRCValid, allAUPRCTest, allEpochs, allMsTimes, allSTimes,
                            allMTimes, allHTimes, defaultClasses, allAUPRCClasses, hierarchyType);
                }
            } else {
                for (int nRun = 1; nRun <= numRuns; nRun++) {
                    HMC_LMLP_Rprop.hmcRpropTrain(arrayArchitecture, numEpochs, datasetTrain, datasetValid, datasetTest, indexesLevels,
                            numLevels, thresholdValues, namesAttributes, nameDatasetValid.get(numData),
                            nameDatasetTest.get(numData), errChoice, nRun, printPredictions, increaseFactor, decreaseFactor,
                            initialDelta, maxDelta, minDelta, weightDecay, learningAlgorithm,
                            thresholdReductionFactor, multilabel, allAUPRCValid, allAUPRCTest, allEpochs, allMsTimes, allSTimes,
                            allMTimes, allHTimes, defaultClasses, allAUPRCClasses, hierarchyType);
                }
            }

            //Calculate means and sd of all runs of the algorithm
            double[] meanSdAUPRCValid = FunctionsCommon.calculateMeanSd(allAUPRCValid);
            double[] meanSdAUPRCTest = FunctionsCommon.calculateMeanSd(allAUPRCTest);
            double[] meanMsTimes = FunctionsCommon.calculateMeanSd(allMsTimes);
            double[] meanSTimes = FunctionsCommon.calculateMeanSd(allSTimes);
            double[] meanMTimes = FunctionsCommon.calculateMeanSd(allMTimes);
            double[] meanHTimes = FunctionsCommon.calculateMeanSd(allHTimes);
            double[] meanEpochs = FunctionsCommon.calculateMeanSd(allEpochs);

            //Each class separatelly
            FunctionsCommon.calculateMeansAUPRCClasses(meansAllAUPRCClasses, allAUPRCClasses);

            //Save the results
            FunctionsCommon.saveResults(nameDatasetTest.get(numData), learningAlgorithm, errChoice,
                    meanSdAUPRCValid, meanSdAUPRCTest, meanMsTimes, meanSTimes, meanMTimes, meanHTimes, meanEpochs);

            FunctionsCommon.saveMeansAUPRCClasses(meansAllAUPRCClasses, nameDatasetTest.get(numData), indexesLevels,
                    numLevels, namesAttributes, errChoice, learningAlgorithm, datasetTest.size());
        }
    }
}
