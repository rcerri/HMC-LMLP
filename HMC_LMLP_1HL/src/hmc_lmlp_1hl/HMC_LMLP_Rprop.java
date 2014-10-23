/*
 * Train the neural network with resilient back-propagation algorithm
 * Not object oriented, although written in Java.
 * That's bad, I know! What a pity! But it works anyway...
 * Absolutely no guarantees or warranties are made concerning the suitability,
 * correctness, or any other aspect of this program. Any use is at your own risk.
 */
package hmc_lmlp_1hl;

import java.util.*;

/**
 *
 * @author Ricardo Cerri
 *
 */
public class HMC_LMLP_Rprop {

    /* =====================================================================================
     * Train of the neural network with resilient back-propagation
     *======================================================================================*/
    static void hmcRpropTrain(ArrayList<Integer[]> arrayArchitecture,
            ArrayList<Integer> numEpochs,
            ArrayList<ArrayList<Double>> datasetTrain,
            ArrayList<ArrayList<Double>> datasetValid,
            ArrayList<ArrayList<Double>> datasetTest,
            ArrayList<ArrayList<Integer>> indexesLevels,
            int numLevels,
            ArrayList<Double> thresholdValues,
            ArrayList<String> namesClasses,
            String nameDatasetValid,
            String nameDatasetTest,
            int errChoice, int numRun,
            int printPredictions,
            double increaseFactor,
            double decreaseFactor,
            double initialDelta,
            double maxDelta,
            double minDelta,
            double weightDecay,
            int learningAlgorithm,
            double thresholdReductionFactor,
            int multilabel,
            double[] allAUPRCValid,
            double[] allAUPRCTest,
            int[] allEpochs,
            double[] allMsTimes,
            double[] allSTimes,
            double[] allMTimes,
            double[] allHTimes,
            double[] defaultClasses,
            ArrayList<ArrayList<double[]>> allAUPRCClasses,
            String hierarchyType) {

        System.out.println("\n============================================================");
        System.out.println("                            Run " + numRun);
        System.out.println("============================================================\n");

        Chronometer chron = new Chronometer();
        chron.start();

        //Parameter of the sigmoidal logistic function
        double a = 1.0;

        //Parameter of the sigmoidal hiperbolic tangent function
        //Suggested by Haykin,1999 (LeCun,1989,1993)
        //double b = 1.7159;
        //double c = 2.0/3.0;
        double b = 1.0;
        double c = 1.0;

        //Initialize weights of the neural network
        ArrayList<ArrayList<Double[][]>> initialNeuralNet = FunctionsCommon.initializeWeights(arrayArchitecture);

        //Get initial weights
        ArrayList<ArrayList<Double[][]>> neuralNet = FunctionsCommon.copyWeights(initialNeuralNet, arrayArchitecture);
        ArrayList<ArrayList<Double[][]>> variationsPrevious = FunctionsCommon.copyWeights(initialNeuralNet, arrayArchitecture);

        //Get weights for bias
        ArrayList<ArrayList<ArrayList<Double>>> biasWeights = FunctionsCommon.initializeBiasWeights(arrayArchitecture);
        ArrayList<ArrayList<ArrayList<Double>>> biasVariationsPrevious = FunctionsCommon.getBiasWeights(arrayArchitecture, biasWeights);

        //Initialize deltas
        ArrayList<ArrayList<Double[][]>> deltas = FunctionsCommon.initializeDeltas(arrayArchitecture, initialDelta);
        ArrayList<ArrayList<Double[][]>> deltasPrevious = FunctionsCommon.initializeDeltas(arrayArchitecture, initialDelta);
        ArrayList<ArrayList<ArrayList<Double>>> deltasBias = FunctionsCommon.initializeBiasDeltasGradients(arrayArchitecture, initialDelta);
        ArrayList<ArrayList<ArrayList<Double>>> deltasBiasPrevious = FunctionsCommon.initializeBiasDeltasGradients(arrayArchitecture, initialDelta);

        //Initialize gradients
        ArrayList<ArrayList<Double[][]>> gradientsPrevious = FunctionsCommon.initializeGradients(arrayArchitecture);
        ArrayList<ArrayList<ArrayList<Double>>> gradientsBiasPrevious = FunctionsCommon.initializeBiasDeltasGradients(arrayArchitecture, 0);

        //Parameter to control epochs of training in each level
        int actEpoch = 0;

        //Maximum number of epochs with no increase in the AU.PRC
        int maxEpochs = 10;
        int controlMaxEpochs = 0;

        //Storage the best results
        double bestAUPRC = 0;
        ArrayList<ArrayList<Double[][]>> bestNeuralNet = FunctionsCommon.initializeWeights(arrayArchitecture);
        ArrayList<ArrayList<ArrayList<Double>>> bestBiasWeights = FunctionsCommon.initializeBiasWeights(arrayArchitecture);
        double[] trainingTimes = new double[4];
        double AUPRC = 0;
        ArrayList<ArrayList<Double>> meanSquareErrors = new ArrayList<ArrayList<Double>>();
        int bestEpoch = 0;
        double[] bestFmeasureThreshold = new double[2];
        double[] fmeasureThreshold = new double[2];


        //Store errors in the previous epoch
        double[] previousError = new double[numLevels];

        for (int indexEpoch = 0; indexEpoch < numEpochs.size(); indexEpoch++) {

            //Number of training epochs
            int numberEpochs = numEpochs.get(indexEpoch);

            //Index Momentum and Learning
            int indexML = -2;

            for (int level = 0; level < numLevels; level++) {

                indexML = indexML + 2;

                //Store mean square error at level
                ArrayList<Double> meanSquareErrorLevel = new ArrayList<Double>();

                int numLevel = level + 1;
                System.out.println("Level " + numLevel);

                for (int epoch = actEpoch + 1; epoch <= numberEpochs; epoch++) {

                    //Initialize gradients
                    ArrayList<ArrayList<Double[][]>> gradients = FunctionsCommon.initializeGradients(arrayArchitecture);
                    ArrayList<ArrayList<ArrayList<Double>>> gradientsBias = FunctionsCommon.initializeBiasDeltasGradients(arrayArchitecture, 0);

                    //Sum of square errors over all training instances
                    double sumSquareError = 0;

                    //Randomize data
                    Collections.shuffle(datasetTrain);

                    //Build structure to store the outputs of the instance
                    ArrayList<ArrayList<Double[]>> outputs = FunctionsCommon.buildOutputStructure(arrayArchitecture,indexesLevels);

                    for (int numInInst = 0; numInInst < datasetTrain.size(); numInInst++) {

                        //Training for the first hierarchical level, so uses training data as input
                        //====================================================================================================
                        //Calculate the output resulted from the data input and the first hidden layer
                        FunctionsCommon.outputsDataInputFirstHLayer(arrayArchitecture, datasetTrain, neuralNet, outputs, numInInst, a, biasWeights);
                        //FunctionsCommon.outputsDataInputFirstHLayer(arrayArchitecture,datasetTrain,neuralNet,outputs,numInInst,b,c,biasWeights);

                        //Calculate the output resulted from the first hidden layer and the first level
                        double sumWeights = FunctionsCommon.outputsFirstHLayerFirstLevel(arrayArchitecture, neuralNet, outputs, numInInst, a, biasWeights);
                        //double sumWeights = FunctionsCommon.outputsFirstHLayerFirstLevel(arrayArchitecture,neuralNet,outputs,numInInst,b,c,biasWeights);

                        //Verify if the instance belongs to the corresponding level
                        //Index of classes of the current level
                        int indexFirstClass = indexesLevels.get(level).get(0);
                        int numClassesLast = indexesLevels.get(level).size();
                        int indexLastClass = indexesLevels.get(level).get(numClassesLast - 1);

                        int belongsLevel = FunctionsCommon.verifyIsPredicted(datasetTrain.get(numInInst), indexFirstClass, indexLastClass);

                        if (belongsLevel == 1) {

                            if (level == 0) {

                                //Update the weights for the first level
                                ArrayList<Double> localGradientOutput = new ArrayList<Double>();
                                ArrayList<Double> localGradientHidden = new ArrayList<Double>();

                                //Local gradient for output neuron and square error over the outputs
                                double squareError = FunctionsCommon.localGradientOutputNeuron(localGradientOutput, outputs, indexesLevels,
                                        datasetTrain, numInInst, a, errChoice, weightDecay, sumWeights);
                                //double squareError = FunctionsCommon.localGradientOutputNeuron(localGradientOutput,outputs,indexesLevels,
                                //                                                         datasetTrain,numInInst,b,c,errChoice,weightDecay,sumWeights);

                                sumSquareError += 0.5 * squareError;

                                //Calculate the gradients between first hidden layer and first level layer
                                FunctionsCommon.calculateGradientsFHLayerFLLayer(arrayArchitecture, outputs,
                                        localGradientOutput, gradients, gradientsBias, numInInst);

                                //Local gradient for hidden neuron
                                FunctionsCommon.localGradientHiddenNeuron(localGradientHidden, localGradientOutput, outputs, neuralNet, a);
                                //FunctionsCommon.localGradientHiddenNeuron(localGradientHidden,localGradientOutput,outputs,neuralNet,b,c);

                                //Calculate the gradients between the first input layer and the first hidden layer
                                FunctionsCommon.calculateGradientsFILayerFHLayer(arrayArchitecture, localGradientHidden, datasetTrain, gradients, gradientsBias, numInInst);

                            }//End if(level == 0)
                            //====================================================================================================
                            else {

                                //Starts iterations through all other hierarchical levels
                                for (int actLevel = 1; actLevel <= level; actLevel++) {

                                    //Calculate the output resulted from one level and one hidden layer
                                    FunctionsCommon.outputsOneLevelOneHiddenLayer(arrayArchitecture, outputs, neuralNet,
                                            actLevel, a, biasWeights, datasetTrain, numInInst, indexesLevels);
                                    //FunctionsCommon.outputsOneLevelOneHiddenLayer(arrayArchitecture,outputs,neuralNet,
                                    //                                        actLevel,b,c,biasWeights,datasetTrain,numInInst);                                        

                                    //Calculate the output resulted from hidden layer and one level
                                    sumWeights = FunctionsCommon.outputsOneHiddenLayerOneLevel(arrayArchitecture, outputs, neuralNet, actLevel, a, biasWeights);
                                    //sumWeights = FunctionsCommon.outputsOneHiddenLayerOneLevel(arrayArchitecture,outputs,neuralNet,actLevel,b,c,biasWeights);

                                }//End actLevel

                                //Compares the network output of a level with the real output
                                //Update the weights for the first level
                                ArrayList<Double> localGradientOutput = new ArrayList<Double>();
                                ArrayList<Double> localGradientHidden = new ArrayList<Double>();
                                ArrayList<Double> outputErrors = new ArrayList<Double>();

                                //Compute the error between the real classes and prediction of the network
                                double squareError = FunctionsCommon.errorRealPredicted(outputs, indexesLevels, datasetTrain,
                                        outputErrors, level, numInInst, errChoice, weightDecay, sumWeights);

                                sumSquareError += 0.5 * squareError;

                                //Computes the local gradients and updates weights for the layers corresponding to the actual level
                                //Local gradient for output neuron
                                FunctionsCommon.localGradientOutputActualNeuron(outputs, localGradientOutput, outputErrors, level, a);
                                //FunctionsCommon.localGradientOutputActualNeuron(outputs,localGradientOutput,outputErrors,level,b,c);

                                //Calcultate the outputs between an hidden layer and a level layer
                                FunctionsCommon.calculateGradientHiddenLayerLevelLayer(arrayArchitecture, outputs, localGradientOutput, gradients, gradientsBias, level);

                                //Local gradient for hidden neuron
                                FunctionsCommon.localGradientHiddenNeuron(localGradientHidden, localGradientOutput, outputs, neuralNet, a, level);
                                //FunctionsCommon.localGradientHiddenNeuron(localGradientHidden,localGradientOutput,outputs,neuralNet,b,c,level);

                                //Calculate gradients between an input layer and an hidden layer
                                FunctionsCommon.calculateGradientInputLayerHiddenLayer(arrayArchitecture, localGradientHidden, outputs, gradients, gradientsBias,
                                        datasetTrain, numInInst, level, indexesLevels);

                            }//End else(level == 0)

                        }//End if (belongsLevel == 1)

                    }//End numInInst

                    double currentError = sumSquareError / datasetTrain.size();

                    //Update phase for all weights and biases

                    //Update the weights between an hidden layer and a level layer
                    FunctionsCommon.updateWeightsHiddenLayerLevelLayer(arrayArchitecture, gradients, deltas, deltasPrevious,
                            gradientsPrevious, neuralNet, variationsPrevious, outputs,
                            increaseFactor, decreaseFactor, initialDelta, maxDelta, minDelta, level,
                            gradientsBias, gradientsBiasPrevious, biasWeights, biasVariationsPrevious, deltasBias, currentError, previousError[level]);

                    //Update the weights between an input layer and an hidden layer
                    FunctionsCommon.updateWeightsInputLayerHiddenLayer(arrayArchitecture, gradients, biasWeights, biasVariationsPrevious, deltas,
                            deltasPrevious, gradientsPrevious, neuralNet, variationsPrevious,
                            deltasBias, deltasBiasPrevious, gradientsBias, gradientsBiasPrevious,
                            increaseFactor, decreaseFactor, initialDelta, maxDelta, minDelta, level, currentError, previousError[level]);

                    //Mean square error over all training instances
                    System.out.println("Epoch = " + epoch + " ---> Error = " + currentError);

                    //Store mean square error at level
                    meanSquareErrorLevel.add(currentError);

                    previousError[level] = currentError;

                }//End numEpochs

                //Store mean square error
                meanSquareErrors.add(meanSquareErrorLevel);

            }//End numLevels

            //Save training time
            chron.stop();

            //Test the trained neural network in valid and test datasets
            //if (multilabel == 1) {

            AUPRC = HMC_LMLP_Test.hmcValid(neuralNet, arrayArchitecture, datasetValid, indexesLevels, numLevels,
                    thresholdValues, a, b, c, namesClasses, biasWeights, thresholdReductionFactor, defaultClasses, fmeasureThreshold, multilabel,hierarchyType);

            System.out.println("\tBest AU.PRC valid = " + bestAUPRC);
            System.out.println("\tCurrent AU.PRC valid = " + AUPRC);

            //Keep the best AUPRC and Neural Net
            if (AUPRC > bestAUPRC) {
                controlMaxEpochs = 0;
                bestAUPRC = AUPRC;
                bestEpoch = numberEpochs;
                bestNeuralNet = FunctionsCommon.copyNeuralNet(bestNeuralNet, neuralNet, arrayArchitecture);
                bestBiasWeights = FunctionsCommon.copyBiasWeights(biasWeights, arrayArchitecture);

                trainingTimes[0] = chron.time();
                trainingTimes[1] = chron.stime();
                trainingTimes[2] = chron.mtime();
                trainingTimes[3] = chron.htime();

                bestFmeasureThreshold[0] = fmeasureThreshold[0];
                bestFmeasureThreshold[1] = fmeasureThreshold[1];
            } else {
                controlMaxEpochs++;
                if (controlMaxEpochs == maxEpochs) {
                    break;
                }
            }

            /*} else {

             FunctionsSingleLabel.printTime(chron.time(), chron.stime(), chron.mtime(), chron.htime(),
             numberEpochs, errChoice, nameDatasetTest, "Training_Time.txt", 1, numRun, learningAlgorithm);

             HMC_LMLP_Test.hmcTest(neuralNet, arrayArchitecture, datasetValid, indexesLevels, numLevels,
             meanSquareErrors, a, b, c, namesClasses, numberEpochs,
             nameDatasetValid, actEpoch, errChoice, biasWeights, numRun, printPredictions,
             learningAlgorithm, multilabel);

             HMC_LMLP_Test.hmcTest(neuralNet, arrayArchitecture, datasetTest, indexesLevels, numLevels,
             meanSquareErrors, a, b, c, namesClasses, numberEpochs,
             nameDatasetTest, actEpoch, errChoice, biasWeights, numRun, printPredictions,
             learningAlgorithm, multilabel);
             }*/

            actEpoch = numberEpochs;

            //Resume training time
            chron.resume();

        }//End numberEpochs

        //Stop training time
        chron.stop();

        System.out.println("Testing best neural network and saving results...");

        //Test the best neural network
        AUPRC = HMC_LMLP_Test.hmcTest(bestNeuralNet, arrayArchitecture, datasetTest, indexesLevels, numLevels,
                thresholdValues, meanSquareErrors, a, b, c, namesClasses, bestEpoch,
                nameDatasetTest, errChoice, bestBiasWeights, numRun, printPredictions,
                learningAlgorithm, thresholdReductionFactor, defaultClasses, bestFmeasureThreshold, allAUPRCClasses, multilabel,hierarchyType);

        //Save the best results
        FunctionsCommon.saveResults(AUPRC, bestAUPRC, bestEpoch, bestNeuralNet, bestBiasWeights,
                trainingTimes, errChoice, learningAlgorithm,
                nameDatasetTest, numRun, meanSquareErrors, numLevels);

        allAUPRCValid[numRun - 1] = bestAUPRC;
        allAUPRCTest[numRun - 1] = AUPRC;
        allEpochs[numRun - 1] = bestEpoch;
        allMsTimes[numRun - 1] = trainingTimes[0];
        allSTimes[numRun - 1] = trainingTimes[1];
        allMTimes[numRun - 1] = trainingTimes[2];
        allHTimes[numRun - 1] = trainingTimes[3];

    }//End hmcRpropTrain
}
