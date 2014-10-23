/*
 * Test the neural network
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
public class HMC_LMLP_Test {

    /* =====================================================================================
     * Test and validation of the trained neural network / Multi-label version with thresholds 
     *======================================================================================*/
    static double hmcValid(ArrayList<ArrayList<Double[][]>> neuralNet,
            ArrayList<Integer[]> arrayArchitecture,
            ArrayList<ArrayList<Double>> datasetTest,
            ArrayList<ArrayList<Integer>> indexesLevels,
            int numLevels,
            ArrayList<Double> thresholdValues,
            double a, double b, double c,
            ArrayList<String> namesAttributes,
            ArrayList<ArrayList<ArrayList<Double>>> biasWeights,
            double thresholdReductionFactor,
            double[] defaultClasses,
            double[] fmeasureThreshold,
            int multilabel,
            String hierarchyType) {

        //Matrix to store the outputs on the test data (values between 0 and 1)
        double[][] matrixOutputsD = new double[datasetTest.size()][datasetTest.get(0).size()];

        for (int numInInst = 0; numInInst < datasetTest.size(); numInInst++) {

            //Build structure to store the outputs
            ArrayList<ArrayList<Double[]>> outputs = FunctionsCommon.buildOutputStructure(arrayArchitecture,indexesLevels);

            //Test for the first hierarchical level
            //====================================================================================================
            //Calculate the output resulted from the data input and the first hidden layer
            //Use logistic in hidden layers
            FunctionsCommon.outputsDataInputFirstHLayer(arrayArchitecture, datasetTest, neuralNet, outputs, numInInst, a, biasWeights);
            //Use hiperbolic tangent in hidden layers
            //FunctionsCommon.outputsDataInputFirstHLayer(arrayArchitecture, datasetTest, neuralNet, outputs, numInInst, b, c, biasWeights);


            //Calculate the output resulted from the first hidden layer and the first level
            //Use logistic in output layers
            FunctionsMultiLabel.outputsFirstHLayerFirstLevel(arrayArchitecture, neuralNet, outputs, numInInst,
                    matrixOutputsD, indexesLevels, a, biasWeights);
            //Use hiperbolic tangent in output layers
            //FunctionsMultiLabel.outputsFirstHLayerFirstLevel(arrayArchitecture, neuralNet, outputs, numInInst,
            //        matrixOutputsD, indexesLevels, b, c, biasWeights);
            //====================================================================================================

            //Starts iterations through all other hierarchical levels
            //====================================================================================================
            for (int actLevel = 1; actLevel < numLevels; actLevel++) {

                //Calculate the output resulted from one level and one hidden layer
                //Use logistic in hidden layers
                FunctionsCommon.outputsOneLevelOneHiddenLayer(arrayArchitecture, outputs, neuralNet, actLevel, a,
                        biasWeights, datasetTest, numInInst);
                //Use hiperbolic tangent in hidden layers
                //FunctionsCommon.outputsOneLevelOneHiddenLayer(arrayArchitecture, outputs, neuralNet, actLevel, b, c,
                //        biasWeights, datasetTest, numInInst);      

                //Calculate the output resulted from hidden layer and one level
                //Use logistic in output layers
                FunctionsMultiLabel.outputsOneHiddenLayerOneLevel(arrayArchitecture, outputs, neuralNet, matrixOutputsD,
                        indexesLevels, actLevel, a, numInInst, biasWeights);
                //Use hiperbolic tangent in output layers
                //FunctionsMultiLabel.outputsOneHiddenLayerOneLevel(arrayArchitecture, outputs, neuralNet, matrixOutputsD,
                //        indexesLevels, actLevel, b, c, numInInst, biasWeights);

            }//End actLevel
            //====================================================================================================

        }//End numInInst

        //Store precision and recall values
        ArrayList<double[][]> valuesPrecisionRecall = new ArrayList<double[][]>();

        //Get the highest outputs in the single-label case
        if (multilabel == 0) {
            matrixOutputsD = FunctionsSingleLabel.getHigherProbabilities(matrixOutputsD, numLevels, indexesLevels, namesAttributes);
        }

        for (int indexThres = 0; indexThres < thresholdValues.size(); indexThres++) {

            //Matrix to store the outputs on the test data after applying thresholds
            int[][] matrixOutputs = new int[datasetTest.size()][datasetTest.get(0).size()];

            //Threshold values used
            double threshold = thresholdValues.get(indexThres) / 100;

            //Apply thresholds in outputs
            //FunctionsMultiLabel.applyThresholds(matrixOutputs,matrixOutputsD,datasetTest,indexesLevels,
            //                                    threshold,thresholdReductionFactor,numLevels,defaultClasses,fmeasureThreshold[1]);
            //Comment this and use above if you want that all classes have at least one prediction,
            //independent of the threshold used
            FunctionsMultiLabel.applyThresholds(matrixOutputs, matrixOutputsD, datasetTest, indexesLevels,
                    threshold, thresholdReductionFactor, numLevels);

            //Verify and correct possible inconsistencies in the predictions
            if (threshold > 0) {
                matrixOutputs = FunctionsCommon.verifyInconsistencies(matrixOutputs, indexesLevels, namesAttributes, hierarchyType);
            }

            //Hierarchical Precision and Recall evaluation metrics
            double[][] evalResults = FunctionsMultiLabel.evaluationPrecRec(datasetTest, matrixOutputs, indexesLevels, numLevels);
            valuesPrecisionRecall.add(evalResults);

            //Calculate fmeasure for current threshold
            double fMeasure = FunctionsCommon.calculateFmeasure(evalResults[0][numLevels - 1], evalResults[1][numLevels - 1]);

            if (fMeasure > fmeasureThreshold[0]) {
                fmeasureThreshold[0] = fMeasure;
                fmeasureThreshold[1] = threshold;
            }

        }//End threshold

        //Calculate AU(PRC)
        double AUPRC = FunctionsMultiLabel.calculateAUPRC(valuesPrecisionRecall, numLevels);

        return AUPRC;

    }

    static double hmcTest(ArrayList<ArrayList<Double[][]>> neuralNet,
            ArrayList<Integer[]> arrayArchitecture,
            ArrayList<ArrayList<Double>> datasetTest,
            ArrayList<ArrayList<Integer>> indexesLevels,
            int numLevels,
            ArrayList<Double> thresholdValues,
            ArrayList<ArrayList<Double>> meanSquareErrors,
            double a, double b, double c,
            ArrayList<String> namesAttributes,
            int numberEpochs,
            String nameDatasetTest,
            int errChoice,
            ArrayList<ArrayList<ArrayList<Double>>> biasWeights,
            int numRun,
            int printPredictions,
            int learningAlgorithm,
            double thresholdReductionFactor,
            double[] defaultClasses,
            double[] bestFmeasureThreshold,
            ArrayList<ArrayList<double[]>> allAUPRCClasses,
            int multilabel,
            String hierarchyType) {

        //Matrix to store the outputs on the test data (values between 0 and 1)
        double[][] matrixOutputsD = new double[datasetTest.size()][datasetTest.get(0).size()];

        for (int numInInst = 0; numInInst < datasetTest.size(); numInInst++) {

            //Build structure to store the outputs
            ArrayList<ArrayList<Double[]>> outputs = FunctionsCommon.buildOutputStructure(arrayArchitecture,indexesLevels);

            //Test for the first hierarchical level
            //====================================================================================================
            //Calculate the output resulted from the data input and the first hidden layer
            //Use logistic in hidden layers
            FunctionsCommon.outputsDataInputFirstHLayer(arrayArchitecture, datasetTest, neuralNet, outputs, numInInst, a, biasWeights);
            //Use hiperbolic tangent in hidden layers
            //FunctionsCommon.outputsDataInputFirstHLayer(arrayArchitecture, datasetTest, neuralNet, outputs, numInInst, b, c, biasWeights);


            //Calculate the output resulted from the first hidden layer and the first level
            //Use logistic in output layers
            FunctionsMultiLabel.outputsFirstHLayerFirstLevel(arrayArchitecture, neuralNet, outputs, numInInst,
                    matrixOutputsD, indexesLevels, a, biasWeights);
            //Use hiperbolic tangent in output layers
            //FunctionsMultiLabel.outputsFirstHLayerFirstLevel(arrayArchitecture, neuralNet, outputs, numInInst,
            //        matrixOutputsD, indexesLevels, b, c, biasWeights);
            //====================================================================================================

            //Starts iterations through all other hierarchical levels
            //====================================================================================================
            for (int actLevel = 1; actLevel < numLevels; actLevel++) {

                //Calculate the output resulted from one level and one hidden layer
                //Use logistic in hidden layers
                FunctionsCommon.outputsOneLevelOneHiddenLayer(arrayArchitecture, outputs, neuralNet, actLevel, a,
                        biasWeights, datasetTest, numInInst);
                //Use hiperbolic tangent in hidden layers
                //FunctionsCommon.outputsOneLevelOneHiddenLayer(arrayArchitecture, outputs, neuralNet, actLevel, b, c,
                //        biasWeights, datasetTest, numInInst);      

                //Calculate the output resulted from hidden layer and one level
                //Use logistic in output layers
                FunctionsMultiLabel.outputsOneHiddenLayerOneLevel(arrayArchitecture, outputs, neuralNet, matrixOutputsD,
                        indexesLevels, actLevel, a, numInInst, biasWeights);
                //Use hiperbolic tangent in output layers
                //FunctionsMultiLabel.outputsOneHiddenLayerOneLevel(arrayArchitecture, outputs, neuralNet, matrixOutputsD,
                //        indexesLevels, actLevel, b, c, numInInst, biasWeights);

            }//End actLevel
            //====================================================================================================

        }//End numInInst

        //Get the highest outputs in the single-label case
        if (multilabel == 0) {
            matrixOutputsD = FunctionsSingleLabel.getHigherProbabilities(matrixOutputsD, numLevels, indexesLevels, namesAttributes);

            //Lets calculate other evaluation measures for single-label problems
            //F-measure per level
            double[] fmeasureLevels = FunctionsSingleLabel.evaluationFmeasure(matrixOutputsD, datasetTest, indexesLevels, 0, 1, numLevels, multilabel);

            //Save the fmeasures for this run
            FunctionsSingleLabel.saveFmeasureRun(fmeasureLevels, errChoice, learningAlgorithm, nameDatasetTest, numRun, numLevels);
        }

        //Save the real number (predictions)
        FunctionsCommon.savePredictions(nameDatasetTest, matrixOutputsD, indexesLevels,
                numLevels, errChoice, numRun, learningAlgorithm);

        ArrayList<double[][]> valuesPrecisionRecall = new ArrayList<double[][]>();
        ArrayList<ArrayList<double[]>> valuesPrecisionRecallClasses = new ArrayList<ArrayList<double[]>>();

        for (int indexThres = 0; indexThres < thresholdValues.size(); indexThres++) {

            //Matrix to store the outputs on the test data after applying thresholds
            int[][] matrixOutputs = new int[datasetTest.size()][datasetTest.get(0).size()];

            //Threshold values used
            double threshold = thresholdValues.get(indexThres) / 100;

            //Apply thresholds in outputs
            //FunctionsMultiLabel.applyThresholds(matrixOutputs,matrixOutputsD,datasetTest,indexesLevels,
            //                                    threshold,thresholdReductionFactor,numLevels,defaultClasses,bestFmeasureThreshold[1]);
            //Comment this and use above if you want that all classes have at least one prediction,
            //independent of the threshold used
            FunctionsMultiLabel.applyThresholds(matrixOutputs, matrixOutputsD, datasetTest, indexesLevels,
                    threshold, thresholdReductionFactor, numLevels);

            //Verify and correct possible inconsistencies in the predictions
            if (threshold > 0) {
                matrixOutputs = FunctionsCommon.verifyInconsistencies(matrixOutputs, indexesLevels, namesAttributes, hierarchyType);
            }
            //Hierarchical Precision and Recall evaluation metrics
            double[][] evalResults = FunctionsMultiLabel.evaluationPrecRec(datasetTest, matrixOutputs, indexesLevels, numLevels);
            valuesPrecisionRecall.add(evalResults);

            //Hierarchical Precision and Recall evaluation for each class individually
            ArrayList<double[]> evalResultsClasses = FunctionsMultiLabel.evaluationPrecRecClasses(datasetTest, matrixOutputs, indexesLevels, numLevels);
            valuesPrecisionRecallClasses.add(evalResultsClasses);

            //Save results
            FunctionsCommon.saveResults(nameDatasetTest, thresholdValues.get(indexThres),
             matrixOutputs, matrixOutputsD, evalResults, indexesLevels,
             numLevels, errChoice, numRun, printPredictions,
             namesAttributes, learningAlgorithm, evalResultsClasses);

        }//End threshold

        //Calculate overall AU(PRC)
        double AUPRC = FunctionsMultiLabel.calculateAUPRC(valuesPrecisionRecall, numLevels, errChoice,
                learningAlgorithm, nameDatasetTest, numRun);

        //Calculate AU(PRC) of each class and save results
        ArrayList<double[]> AUPRCClasses = FunctionsMultiLabel.calculateAUPRCClasses(valuesPrecisionRecallClasses,
                numLevels, indexesLevels, namesAttributes, errChoice,
                learningAlgorithm, nameDatasetTest, numRun);

        FunctionsCommon.saveAUPRCClasses(AUPRCClasses, indexesLevels,
                nameDatasetTest, namesAttributes, numRun, numLevels, errChoice, learningAlgorithm);

        allAUPRCClasses.add(AUPRCClasses);

        return AUPRC;

    }

    static double hmcValidTrueLabels(ArrayList<ArrayList<Double[][]>> neuralNet,
            ArrayList<Integer[]> arrayArchitecture,
            ArrayList<ArrayList<Double>> datasetTest,
            ArrayList<ArrayList<Integer>> indexesLevels,
            int numLevels,
            ArrayList<Double> thresholdValues,
            double a, double b, double c,
            ArrayList<String> namesAttributes,
            ArrayList<ArrayList<ArrayList<Double>>> biasWeights,
            double thresholdReductionFactor,
            double[] defaultClasses,
            double[] fmeasureThreshold,
            int multilabel,
            String hierarchyType) {

        //Store precision and recall values
        ArrayList<double[][]> valuesPrecisionRecall = new ArrayList<double[][]>();

        for (int indexThres = 0; indexThres < thresholdValues.size(); indexThres++) {

            //Threshold values used
            double threshold = thresholdValues.get(indexThres) / 100;

            //Matrix to store the outputs on the test data (values between 0 and 1)
            double[][] matrixOutputsD = new double[datasetTest.size()][datasetTest.get(0).size()];

            for (int numInInst = 0; numInInst < datasetTest.size(); numInInst++) {

                //Build structure to store the outputs
                ArrayList<ArrayList<Double[]>> outputs = FunctionsCommon.buildOutputStructure(arrayArchitecture,indexesLevels);

                //Test for the first hierarchical level
                //====================================================================================================
                //Calculate the output resulted from the data input and the first hidden layer
                //Use logistic in hidden layers
                FunctionsCommon.outputsDataInputFirstHLayer(arrayArchitecture, datasetTest, neuralNet, outputs, numInInst, a, biasWeights);
                //Use hiperbolic tangent in hidden layers
                //FunctionsCommon.outputsDataInputFirstHLayer(arrayArchitecture, datasetTest, neuralNet, outputs, numInInst, b, c, biasWeights);


                //Calculate the output resulted from the first hidden layer and the first level
                //Use logistic in output layers
                FunctionsMultiLabel.outputsFirstHLayerFirstLevel(arrayArchitecture, neuralNet, outputs, numInInst,
                        matrixOutputsD, indexesLevels, a, biasWeights);
                //Use hiperbolic tangent in output layers
                //FunctionsMultiLabel.outputsFirstHLayerFirstLevel(arrayArchitecture, neuralNet, outputs, numInInst,
                //        matrixOutputsD, indexesLevels, b, c, biasWeights);
                //====================================================================================================

                //Starts iterations through all other hierarchical levels
                //====================================================================================================
                for (int actLevel = 1; actLevel < numLevels; actLevel++) {

                    //Calculate the output resulted from one level and one hidden layer
                    //Use logistic in hidden layers
                    FunctionsCommon.outputsOneLevelOneHiddenLayer(arrayArchitecture, outputs, neuralNet, actLevel, a,
                            biasWeights, datasetTest, numInInst, threshold, thresholdReductionFactor);
                    //Use hiperbolic tangent in hidden layers
                    //FunctionsCommon.outputsOneLevelOneHiddenLayer(arrayArchitecture, outputs, neuralNet, actLevel, b, c,
                    //        biasWeights, datasetTest, numInInst);      

                    //Calculate the output resulted from hidden layer and one level
                    //Use logistic in output layers
                    FunctionsMultiLabel.outputsOneHiddenLayerOneLevel(arrayArchitecture, outputs, neuralNet, matrixOutputsD,
                            indexesLevels, actLevel, a, numInInst, biasWeights);
                    //Use hiperbolic tangent in output layers
                    //FunctionsMultiLabel.outputsOneHiddenLayerOneLevel(arrayArchitecture, outputs, neuralNet, matrixOutputsD,
                    //        indexesLevels, actLevel, b, c, numInInst, biasWeights);

                }//End actLevel
                //====================================================================================================

            }//End numInInst

            //Get the highest outputs in the single-label case
            if (multilabel == 0) {
                matrixOutputsD = FunctionsSingleLabel.getHigherProbabilities(matrixOutputsD, numLevels, indexesLevels, namesAttributes);
            }

            //Matrix to store the outputs on the test data after applying thresholds
            int[][] matrixOutputs = new int[datasetTest.size()][datasetTest.get(0).size()];

            //Apply thresholds in outputs
            //FunctionsMultiLabel.applyThresholds(matrixOutputs,matrixOutputsD,datasetTest,indexesLevels,
            //                                    threshold,thresholdReductionFactor,numLevels,defaultClasses,fmeasureThreshold[1]);
            //Comment this and use above if you want that all classes have at least one prediction,
            //independent of the threshold used
            FunctionsMultiLabel.applyThresholds(matrixOutputs, matrixOutputsD, datasetTest, indexesLevels,
                    threshold, thresholdReductionFactor, numLevels);

            //Verify and correct possible inconsistencies in the predictions
            if (threshold > 0) {
                matrixOutputs = FunctionsCommon.verifyInconsistencies(matrixOutputs, indexesLevels, namesAttributes, hierarchyType);
            }

            //Hierarchical Precision and Recall evaluation metrics
            double[][] evalResults = FunctionsMultiLabel.evaluationPrecRec(datasetTest, matrixOutputs, indexesLevels, numLevels);
            valuesPrecisionRecall.add(evalResults);

            //Calculate fmeasure for current threshold
            /*double fMeasure = FunctionsCommon.calculateFmeasure(evalResults[0][numLevels - 1], evalResults[1][numLevels - 1]);

             if (fMeasure > fmeasureThreshold[0]) {
             fmeasureThreshold[0] = fMeasure;
             fmeasureThreshold[1] = threshold;
             }*/

        }//End threshold

        //Calculate AU(PRC)
        double AUPRC = FunctionsMultiLabel.calculateAUPRC(valuesPrecisionRecall, numLevels);

        return AUPRC;

    }

    static double hmcTestTrueLabels(ArrayList<ArrayList<Double[][]>> neuralNet,
            ArrayList<Integer[]> arrayArchitecture,
            ArrayList<ArrayList<Double>> datasetTest,
            ArrayList<ArrayList<Integer>> indexesLevels,
            int numLevels,
            ArrayList<Double> thresholdValues,
            ArrayList<ArrayList<Double>> meanSquareErrors,
            double a, double b, double c,
            ArrayList<String> namesAttributes,
            int numberEpochs,
            String nameDatasetTest,
            int errChoice,
            ArrayList<ArrayList<ArrayList<Double>>> biasWeights,
            int numRun,
            int printPredictions,
            int learningAlgorithm,
            double thresholdReductionFactor,
            double[] defaultClasses,
            double[] bestFmeasureThreshold,
            ArrayList<ArrayList<double[]>> allAUPRCClasses,
            int multilabel,
            String hierarchyType) {

        ArrayList<double[][]> valuesPrecisionRecall = new ArrayList<double[][]>();
        ArrayList<ArrayList<double[]>> valuesPrecisionRecallClasses = new ArrayList<ArrayList<double[]>>();

        for (int indexThres = 0; indexThres < thresholdValues.size(); indexThres++) {

            //Threshold values used
            double threshold = thresholdValues.get(indexThres) / 100;

            //Matrix to store the outputs on the test data (values between 0 and 1)
            double[][] matrixOutputsD = new double[datasetTest.size()][datasetTest.get(0).size()];

            for (int numInInst = 0; numInInst < datasetTest.size(); numInInst++) {

                //Build structure to store the outputs
                ArrayList<ArrayList<Double[]>> outputs = FunctionsCommon.buildOutputStructure(arrayArchitecture,indexesLevels);

                //Test for the first hierarchical level
                //====================================================================================================
                //Calculate the output resulted from the data input and the first hidden layer
                //Use logistic in hidden layers
                FunctionsCommon.outputsDataInputFirstHLayer(arrayArchitecture, datasetTest, neuralNet, outputs, numInInst, a, biasWeights);
                //Use hiperbolic tangent in hidden layers
                //FunctionsCommon.outputsDataInputFirstHLayer(arrayArchitecture, datasetTest, neuralNet, outputs, numInInst, b, c, biasWeights);


                //Calculate the output resulted from the first hidden layer and the first level
                //Use logistic in output layers
                FunctionsMultiLabel.outputsFirstHLayerFirstLevel(arrayArchitecture, neuralNet, outputs, numInInst,
                        matrixOutputsD, indexesLevels, a, biasWeights);
                //Use hiperbolic tangent in output layers
                //FunctionsMultiLabel.outputsFirstHLayerFirstLevel(arrayArchitecture, neuralNet, outputs, numInInst,
                //        matrixOutputsD, indexesLevels, b, c, biasWeights);
                //====================================================================================================

                //Starts iterations through all other hierarchical levels
                //====================================================================================================
                for (int actLevel = 1; actLevel < numLevels; actLevel++) {

                    //Calculate the output resulted from one level and one hidden layer
                    //Use logistic in hidden layers
                    FunctionsCommon.outputsOneLevelOneHiddenLayer(arrayArchitecture, outputs, neuralNet, actLevel, a,
                            biasWeights, datasetTest, numInInst, threshold, thresholdReductionFactor);
                    //Use hiperbolic tangent in hidden layers
                    //FunctionsCommon.outputsOneLevelOneHiddenLayer(arrayArchitecture, outputs, neuralNet, actLevel, b, c,
                    //        biasWeights, datasetTest, numInInst);      

                    //Calculate the output resulted from hidden layer and one level
                    //Use logistic in output layers
                    FunctionsMultiLabel.outputsOneHiddenLayerOneLevel(arrayArchitecture, outputs, neuralNet, matrixOutputsD,
                            indexesLevels, actLevel, a, numInInst, biasWeights);
                    //Use hiperbolic tangent in output layers
                    //FunctionsMultiLabel.outputsOneHiddenLayerOneLevel(arrayArchitecture, outputs, neuralNet, matrixOutputsD,
                    //        indexesLevels, actLevel, b, c, numInInst, biasWeights);

                }//End actLevel
                //====================================================================================================

            }//End numInInst

            //Get the highest outputs in the single-label case
            if (multilabel == 0) {
                matrixOutputsD = FunctionsSingleLabel.getHigherProbabilities(matrixOutputsD, numLevels, indexesLevels, namesAttributes);

                //Lets calculate other evaluation measures for single-label problems
                //F-measure per level
                double[] fmeasureLevels = FunctionsSingleLabel.evaluationFmeasure(matrixOutputsD, datasetTest, indexesLevels, 0, 1, numLevels, multilabel);

                //Save the fmeasures for this run
                FunctionsSingleLabel.saveFmeasureRun(fmeasureLevels, errChoice, learningAlgorithm, nameDatasetTest, numRun, numLevels);
            }

            //Save the real number (predictions)
            FunctionsCommon.savePredictions(nameDatasetTest, matrixOutputsD, indexesLevels,
                    numLevels, errChoice, numRun, learningAlgorithm);

            //Matrix to store the outputs on the test data after applying thresholds
            int[][] matrixOutputs = new int[datasetTest.size()][datasetTest.get(0).size()];

            //Apply thresholds in outputs
            //FunctionsMultiLabel.applyThresholds(matrixOutputs,matrixOutputsD,datasetTest,indexesLevels,
            //                                    threshold,thresholdReductionFactor,numLevels,defaultClasses,bestFmeasureThreshold[1]);
            //Comment this and use above if you want that all classes have at least one prediction,
            //independent of the threshold used
            FunctionsMultiLabel.applyThresholds(matrixOutputs, matrixOutputsD, datasetTest, indexesLevels,
                    threshold, thresholdReductionFactor, numLevels);

            //Verify and correct possible inconsistencies in the predictions
            if (threshold > 0) {
                matrixOutputs = FunctionsCommon.verifyInconsistencies(matrixOutputs, indexesLevels, namesAttributes, hierarchyType);
            }
            //Hierarchical Precision and Recall evaluation metrics
            double[][] evalResults = FunctionsMultiLabel.evaluationPrecRec(datasetTest, matrixOutputs, indexesLevels, numLevels);
            valuesPrecisionRecall.add(evalResults);

            //Hierarchical Precision and Recall evaluation for each class individually
            ArrayList<double[]> evalResultsClasses = FunctionsMultiLabel.evaluationPrecRecClasses(datasetTest, matrixOutputs, indexesLevels, numLevels);
            valuesPrecisionRecallClasses.add(evalResultsClasses);

            //Save results
            FunctionsCommon.saveResults(nameDatasetTest, thresholdValues.get(indexThres),
             matrixOutputs, matrixOutputsD, evalResults, indexesLevels,
             numLevels, errChoice, numRun, printPredictions,
             namesAttributes, learningAlgorithm, evalResultsClasses);

        }//End threshold

        //Calculate overall AU(PRC)
        double AUPRC = FunctionsMultiLabel.calculateAUPRC(valuesPrecisionRecall, numLevels, errChoice,
                learningAlgorithm, nameDatasetTest, numRun);

        //Calculate AU(PRC) of each class and save results
        ArrayList<double[]> AUPRCClasses = FunctionsMultiLabel.calculateAUPRCClasses(valuesPrecisionRecallClasses,
                numLevels, indexesLevels, namesAttributes, errChoice,
                learningAlgorithm, nameDatasetTest, numRun);

        FunctionsCommon.saveAUPRCClasses(AUPRCClasses, indexesLevels,
                nameDatasetTest, namesAttributes, numRun, numLevels, errChoice, learningAlgorithm);

        allAUPRCClasses.add(AUPRCClasses);

        return AUPRC;

    }


    /* =====================================================================================
     * Test of the trained neural network / Single-label version without thresholds 
     *======================================================================================*/
    static void hmcTest(ArrayList<ArrayList<Double[][]>> neuralNet,
            ArrayList<Integer[]> arrayArchitecture,
            ArrayList<ArrayList<Double>> datasetTest,
            ArrayList<ArrayList<Integer>> indexesLevels,
            int numLevels,
            ArrayList<ArrayList<Double>> meanSquareErrors,
            double a,
            double b,
            double c,
            ArrayList<String> namesAttributes,
            int numberEpochs,
            String nameDatasetTest,
            int actEpoch,
            int errChoice,
            ArrayList<ArrayList<ArrayList<Double>>> biasWeights,
            int numRun,
            int printPredictions,
            int learningAlgorithm,
            int multilabel,
            String hierarchyType) {

        Chronometer chron = new Chronometer();
        chron.start();

        //Matrix to store the outputs on the test data
        int[][] matrixOutputs = new int[datasetTest.size()][datasetTest.get(0).size()];
        double[][] matrixOutputsD = new double[datasetTest.size()][datasetTest.get(0).size()];

        for (int numInInst = 0; numInInst < datasetTest.size(); numInInst++) {

            //Build structure to store the outputs
            ArrayList<ArrayList<Double[]>> outputs = FunctionsCommon.buildOutputStructure(arrayArchitecture,indexesLevels);

            //Test for the first hierarchical level
            //====================================================================================================
            //Calculate the output resulted from the data input and the first hidden layer
            FunctionsCommon.outputsDataInputFirstHLayer(arrayArchitecture, datasetTest, neuralNet, outputs, numInInst, a, biasWeights);
            //FunctionsCommon.outputsDataInputFirstHLayer(arrayArchitecture, datasetTest, neuralNet, outputs, numInInst, b, c, biasWeights);

            //Calculate the output resulted from the first hidden layer and the first level
            FunctionsSingleLabel.outputsFirstHLayerFirstLevel(arrayArchitecture, neuralNet, outputs, numInInst, matrixOutputs,
                    matrixOutputsD, indexesLevels, a, biasWeights);
            //  FunctionsSingleLabel.outputsFirstHLayerFirstLevel(arrayArchitecture, neuralNet, outputs, numInInst, matrixOutputs,
            //        matrixOutputsD, indexesLevels, b, c, biasWeights);
            //====================================================================================================

            //Starts iterations through all other hierarchical levels
            //====================================================================================================
            for (int actLevel = 1; actLevel < numLevels; actLevel++) {

                //Calculate the output resulted from one level and one hidden layer
                FunctionsCommon.outputsOneLevelOneHiddenLayer(arrayArchitecture, outputs, neuralNet, actLevel, a,
                        biasWeights, datasetTest, numInInst);
                //FunctionsCommon.outputsOneLevelOneHiddenLayer(arrayArchitecture, outputs, neuralNet, actLevel, b, c,
                //        biasWeights, datasetTest, numInInst);

                //Calculate the output resulted from hidden layer and one level
                FunctionsSingleLabel.outputsOneHiddenLayerOneLevel(arrayArchitecture, outputs, neuralNet, matrixOutputs, matrixOutputsD,
                        indexesLevels, actLevel, a, numInInst, biasWeights);
                //FunctionsSingleLabel.outputsOneHiddenLayerOneLevel(arrayArchitecture, outputs, neuralNet, matrixOutputs, matrixOutputsD,
                //                                        indexesLevels, actLevel, b, c, numInInst, biasWeights);

            }//End actLevel
            //====================================================================================================

        }//End numInInst

        //Print predictions
        //Functions.printPredictions(matrixOutputs,indexesLevels,0);

        //Verify and correct possible inconsistencies in the predictions
        matrixOutputs = FunctionsCommon.verifyInconsistencies(matrixOutputs, indexesLevels, namesAttributes, hierarchyType);

        //Print predictions
        //Functions.printPredictions(matrixOutputs,indexesLevels,1);

        //Hierarchical Precision and Recall evaluation metrics
        double[][] evalResults = FunctionsSingleLabel.evaluationPrecRecAcc(nameDatasetTest, datasetTest, matrixOutputs,
                indexesLevels, numLevels, namesAttributes,
                errChoice, numRun, learningAlgorithm, numberEpochs);

        //Stop test time
        chron.stop();

        //Save results
        FunctionsSingleLabel.saveResults(nameDatasetTest, numberEpochs, matrixOutputs, matrixOutputsD,
                evalResults, indexesLevels, meanSquareErrors, numLevels, actEpoch,
                errChoice, numRun, printPredictions, namesAttributes, learningAlgorithm);

        //Resume test time
        chron.resume();

        //Stop test time
        chron.stop();

        //Save test time
        FunctionsSingleLabel.printTime(chron.time(), chron.stime(), chron.mtime(), chron.htime(),
                numberEpochs, errChoice, nameDatasetTest, "Test_Time.txt", 2, numRun, learningAlgorithm);

    }
}
