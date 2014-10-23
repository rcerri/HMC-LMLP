/*
 * Train the neural network with back-propagation algorithm
 * Not object oriented, although written in Java.
 * That's bad, I know! What a pity! But it works anyway...
 * 
 * Absolutely no guarantees or warranties are made concerning the suitability,
 * correctness, or any other aspect of this program. Any use is at your own risk.
 */
package hmc_lmlp_1hl;

import java.util.*;

/**
 *
 * @author Ricardo Cerri
 */
public class HMC_LMLP_BP {

    /* =====================================================================================
     * Train of the neural network with back-propagation
     *======================================================================================*/
    static void hmcBpTrain(ArrayList<Integer[]> arrayArchitecture,
            ArrayList<Integer> numEpochs,
            ArrayList<ArrayList<Double>> datasetTrain,
            ArrayList<ArrayList<Double>> datasetValid,
            ArrayList<ArrayList<Double>> datasetTest,
            ArrayList<ArrayList<Integer>> indexesLevels,
            int numLevels,
            ArrayList<Double> learningRate,
            ArrayList<Double> momentumConstant,
            ArrayList<Double> thresholdValues,
            ArrayList<String> namesClasses,
            String nameDatasetValid,
            String nameDatasetTest,
            int errChoice, int numRun,
            double weightDecay,
            int printPredictions,
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

                    //Sum of square errors over all training instances
                    double sumSquareError = 0;

                    //Randomize data
                    Collections.shuffle(datasetTrain);

                    for (int numInInst = 0; numInInst < datasetTrain.size(); numInInst++) {

                        //Build structure to store the outputs of the instance
                        ArrayList<ArrayList<Double[]>> outputs = FunctionsCommon.buildOutputStructure(arrayArchitecture,indexesLevels);

                        //Training for the first hierarchical level, so uses training data as input
                        //====================================================================================================
                        //Calculate the output resulted from the data input and the first hidden layer
                        //Use logistic function in hidden layers
                        FunctionsCommon.outputsDataInputFirstHLayer(arrayArchitecture, datasetTrain, neuralNet, outputs, numInInst, a, biasWeights);
                        //Use hiperbolic tangent in hidden layers
                        //FunctionsCommon.outputsDataInputFirstHLayer(arrayArchitecture, datasetTrain, neuralNet, outputs, numInInst, b, c, biasWeights);

                        //Calculate the output resulted from the first hidden layer and the first level
                        //Use logistic function in output layers
                        double sumWeights = FunctionsCommon.outputsFirstHLayerFirstLevel(arrayArchitecture, neuralNet, outputs, numInInst, a, biasWeights);
                        //Use hiperbolic tangent in output layers
                        //double sumWeights = FunctionsCommon.outputsFirstHLayerFirstLevel(arrayArchitecture, neuralNet, outputs, numInInst, b, c, biasWeights);

                        //Verify if the instance belongs to the corresponding level
                        //Index of classes of the current level
                        int indexFirstClass = indexesLevels.get(level).get(0);
                        int numClassesLast = indexesLevels.get(level).size();
                        int indexLastClass = indexesLevels.get(level).get(numClassesLast - 1);

                        //int belongsLevel = FunctionsCommon.verifyIsPredicted(datasetTrain.get(numInInst), indexFirstClass, indexLastClass);

                        //if (belongsLevel == 1) {

                            if (level == 0) {

                                //Update the weights for the first level
                                ArrayList<Double> localGradientOutput = new ArrayList<Double>();
                                ArrayList<Double> localGradientHidden = new ArrayList<Double>();

                                //Local gradient for output neuron and square error over the outputs
                                //Use logistic function in output layers
                                double squareError = FunctionsCommon.localGradientOutputNeuron(localGradientOutput, outputs, indexesLevels,
                                        datasetTrain, numInInst, a, errChoice, weightDecay, sumWeights);
                                //Use hiperbolic tangent in output layers
                                //double squareError = FunctionsCommon.localGradientOutputNeuron(localGradientOutput, outputs, indexesLevels,
                                //        datasetTrain, numInInst, b, c, errChoice, weightDecay, sumWeights);

                                sumSquareError += 0.5 * squareError;

                                //Update the weights between the first hidden layer and the first level layer
                                FunctionsCommon.updateWeightsFHLayerFLLayer(arrayArchitecture, outputs, localGradientOutput,
                                        neuralNet, variationsPrevious, learningRate.get(1),
                                        momentumConstant.get(1), numInInst, weightDecay, biasWeights, biasVariationsPrevious);

                                //Local gradient for hidden neuron
                                //Use logistic function in hidden layers
                                FunctionsCommon.localGradientHiddenNeuron(localGradientHidden, localGradientOutput, outputs, neuralNet, a);
                                //Use hiperbolic tanget function in hidden layers
                                //FunctionsCommon.localGradientHiddenNeuron(localGradientHidden, localGradientOutput, outputs, neuralNet, b, c);

                                //Update the weights between the input layer and the first hidden layer
                                FunctionsCommon.updateWeightsFILayerFHLayer(arrayArchitecture, localGradientHidden, datasetTrain, neuralNet,
                                        variationsPrevious, learningRate.get(0), momentumConstant.get(0), numInInst,
                                        biasWeights, biasVariationsPrevious, weightDecay);

                            }//End if(level == 0)
                            //====================================================================================================
                            else {

                                //Starts iterations through all other hierarchical levels
                                for (int actLevel = 1; actLevel <= level; actLevel++) {
                                    
                                    //Calculate the output resulted from one level and one hidden layer
                                    //Use logistic function in hidden layers
                                    FunctionsCommon.outputsOneLevelOneHiddenLayer(arrayArchitecture, outputs, neuralNet,
                                            actLevel, a, biasWeights, datasetTrain, numInInst, indexesLevels);
                                    //Use hiperbolic tanget function in hidden layers
                                    //FunctionsCommon.outputsOneLevelOneHiddenLayer(arrayArchitecture, outputs, neuralNet,
                                    //        actLevel, b, c, biasWeights, datasetTrain, numInInst);

                                    //Calculate the output resulted from hidden layer and one level
                                    //Use logistic function in output layers
                                    sumWeights = FunctionsCommon.outputsOneHiddenLayerOneLevel(arrayArchitecture, outputs, neuralNet, actLevel, a, biasWeights);
                                    //Use hiperbolic tanget function in output layers
                                    //sumWeights = FunctionsCommon.outputsOneHiddenLayerOneLevel(arrayArchitecture, outputs, neuralNet, actLevel, b, c, biasWeights);

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
                                //Use logistic function in output layers
                                FunctionsCommon.localGradientOutputActualNeuron(outputs, localGradientOutput, outputErrors, level, a);
                                //Use hiperbolic tangent function in output layers
                                //FunctionsCommon.localGradientOutputActualNeuron(outputs, localGradientOutput, outputErrors, level, b, c);


                                //Update the weights between an hidden layer and a level layer
                                FunctionsCommon.updateWeightsHiddenLayerLevelLayer(arrayArchitecture, outputs, localGradientOutput,
                                        neuralNet, variationsPrevious, learningRate.get(indexML+1),
                                        momentumConstant.get(indexML+1), level, numInInst, weightDecay,
                                        biasWeights, biasVariationsPrevious);

                                //Local gradient for hidden neuron
                                //Use logistic function in hidden layers
                                FunctionsCommon.localGradientHiddenNeuron(localGradientHidden, localGradientOutput, outputs, neuralNet, a, level);
                                //Use hiperbolic tangent function in hidden layers
                                //FunctionsCommon.localGradientHiddenNeuron(localGradientHidden, localGradientOutput, outputs, neuralNet, b, c, level);

                                //Update the weights between an input layer and an hidden layer
                                FunctionsCommon.updateWeightsInputLayerHiddenLayer(arrayArchitecture, localGradientHidden, outputs, neuralNet,
                                        variationsPrevious, learningRate.get(indexML), momentumConstant.get(indexML),
                                        level, numInInst, biasWeights, biasVariationsPrevious,
                                        datasetTrain, weightDecay, indexesLevels);

                            }//End else(level == 0)

                        //}//End if (belongsLevel == 1)

                    }//End numInInst

                    //Mean square error over all training instances
                    System.out.println("Epoch = " + epoch + " ---> Error = " + sumSquareError / datasetTrain.size());

                    //Store mean square error at level
                    meanSquareErrorLevel.add(sumSquareError / datasetTrain.size());

                }//End numEpochs

                //Store mean square error
                meanSquareErrors.add(meanSquareErrorLevel);

            }//End numLevels

            //Save training time
            chron.stop();

            AUPRC = HMC_LMLP_Test.hmcValid(neuralNet, arrayArchitecture, datasetValid, indexesLevels, numLevels,
                    thresholdValues, a, b, c, namesClasses, biasWeights, thresholdReductionFactor, defaultClasses,
                    fmeasureThreshold, multilabel,hierarchyType);

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

    }//End hmcBpTrain
}
