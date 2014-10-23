/* Paths to be used in HMC-LMLP to save results
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
public class Paths {

    //Variables to be used
    private String filePath;
    private String fileResults;
    private String fileResultsClasses;
    private String fileMeanSquareErrorsTraining;
    private String filePredictions;
    private String strDirectoryDatasetValid;
    private String strDirectoryDatasetTest;
    private String fileOutput;
    private String strDirectoryPRCurvesValid;
    private String strDirectoryPRCurvesTest;
    private String fileConfusionMatrices;
    private String fileBestResults;
    private String strDirectoryBestResults;
    private String fileBestNeuralNetwork;
    private String strDirectoryAlgorithm;
    private String filePredictionRealNumbers;
    private String fileInterpolations;
    private String fileInterpolationsClasses;
    private String fileBestBiasWeights;
    private String fileBestResultsAUPRCClasses;
    private String fileFmeasureLevels;

    
    /*===========================================================================
     * Set the learning algorithm to be used
     *===========================================================================*/
    public String setLearningAlgorithm(int learningAlgorithm) {

        String dirAlg = "";

        switch (learningAlgorithm) {
            case 1:
                dirAlg = "Backpropagation";
                break;
            case 2:
                dirAlg = "Rprop";
                break;
        }

        return dirAlg;
    }

    /*===========================================================================
     * Set the error measure to be used in training
     *===========================================================================*/
    public String setErrorChoice(int errChoice) {

        String dirErr = "";

        switch (errChoice) {
            case 1:
                dirErr = "ConventionalError";
                break;
            case 2:
                dirErr = "ZhangZhouError";
                break;
        }

        return dirErr;
    }

    /*===========================================================================
     * Set the paths to be used in multi-label version
     *===========================================================================*/
    public void setPathsMultiLabel(String dirAlg, String dirErr,
            String nameDatasetTest, int numRun, double threshold) {

        filePath = "Results/Multilabel/" + dirAlg + "/" + dirErr + "/Run" + numRun + "/" + nameDatasetTest + "/TestResults/" + threshold;
        //String filePathD = "Results/" + dirAlg + "/" + dirErr + "/Run" + numRun + "/" + nameDatasetTest + "/" + numberEpochs + "/";
        //String filePredictions = filePath + "/" + "predictions.txt";
        //String filePredictionsD = filePathD + "/" + "predictionsD.txt";
        fileResults = filePath + "/results.txt";
        fileResultsClasses = filePath + "/resultsClasses.txt";
        filePredictions = filePath + "/predictions.txt";

    }

    public void setPathsMultiLabel(String dirAlg, String dirErr, int numberEpochs,
            String nameDatasetTest, int numRun, String nameOutput) {

        fileOutput = "Results/Multilabel/" + dirAlg + "/" + dirErr + "/Run" + numRun + "/" + nameDatasetTest + "/" + numberEpochs + "/" + nameOutput;
    }

    public void setPathsMultilabel(String dirAlg, String dirErr) {

        strDirectoryAlgorithm = "Results/Multilabel/" + dirAlg + "/" + dirErr;

    }

    public void setPathsMultiLabel(String dirAlg, String dirErr,
            String nameDatasetTest, int nRun) {

        //strDirectoryDatasetValid = "Results/Multilabel/" + dirAlg + "/" + dirErr + "/Run" + nRun + "/" + nameDatasetValid;
        //strDirectoryPRCurvesValid = "Results/Multilabel/" + dirAlg + "/PR_Curves/" + dirErr + "/Run" + nRun + "/" + nameDatasetValid;
        strDirectoryBestResults = "Results/Multilabel/" + dirAlg + "/" + dirErr + "/Run" + nRun + "/" + nameDatasetTest;
        strDirectoryDatasetTest = strDirectoryBestResults + "/TestResults";
        fileBestResults = strDirectoryBestResults + "/results.txt";
        fileBestResultsAUPRCClasses = strDirectoryBestResults + "/resultsAUPRCClasses.txt";
        fileBestNeuralNetwork = strDirectoryBestResults + "/bestNeuralNetModel.ser";
        fileBestBiasWeights = strDirectoryBestResults + "/bestBiasWeights.ser";
        fileMeanSquareErrorsTraining = strDirectoryBestResults + "/meanSquareErrorsTraining.txt";
        filePredictionRealNumbers = strDirectoryDatasetTest + "/predictions.txt";
        //strDirectoryPRCurvesTest = "Results/Multilabel/" + dirAlg + "/PR_Curves/" + dirErr + "/Run" + nRun + "/" + nameDatasetTest;
        fileInterpolations = strDirectoryDatasetTest + "/interpolations.txt";
        fileInterpolationsClasses = strDirectoryDatasetTest + "/interpolationsClasses.txt";
    }

    /*===========================================================================
     * Set the paths to be used in single-label version
     *===========================================================================*/
    public void setPathsSingleLabel(String dirAlg, String dirErr, int numberEpochs,
            String nameDatasetTest, int numRun) {

        filePath = "Results/Singlelabel/" + dirAlg + "/" + dirErr + "/Run" + numRun + "/" + nameDatasetTest + "/" + numberEpochs;
        //String filePathD = "Results/" + dirAlg + "/" + dirErr + "/Run" + numRun + "/" + nameDatasetTest + "/" + numberEpochs + "/";
        //String filePredictions = filePath + "/" + "predictions.txt";
        //String filePredictionsD = filePathD + "/" + "predictionsD.txt";
        fileResults = filePath + "/" + "results.txt";
        fileMeanSquareErrorsTraining = "Results/Singlelabel/" + dirAlg + "/" + dirErr + "/Run" + numRun + "/" + nameDatasetTest + "/" + numberEpochs + "/" + "meanSquareErrorsTraining.txt";
        filePredictions = filePath + "/" + "predictions.txt";
        fileConfusionMatrices = filePath + "/confusionMatrices.ser";
    }

    public void setPathsSingleLabel(String dirAlg, String dirErr, ArrayList<Integer> numberEpochs,
            String nameDatasetValid, String nameDatasetTest, int nRun) {

        strDirectoryDatasetValid = "Results/Singlelabel/" + dirAlg + "/" + dirErr + "/Run" + nRun + "/" + nameDatasetValid;
        strDirectoryDatasetTest = "Results/Singlelabel/" + dirAlg + "/" + dirErr + "/Run" + nRun + "/" + nameDatasetTest;

    }
    
    public void setPaths(String dirAlg, String dirErr,
            String nameDatasetTest, int nRun) {

        //strDirectoryDatasetValid = "Results/Multilabel/" + dirAlg + "/" + dirErr + "/Run" + nRun + "/" + nameDatasetValid;
        //strDirectoryPRCurvesValid = "Results/Multilabel/" + dirAlg + "/PR_Curves/" + dirErr + "/Run" + nRun + "/" + nameDatasetValid;
        strDirectoryBestResults = "Results/" + dirAlg + "/" + dirErr + "/Run" + nRun + "/" + nameDatasetTest;
        strDirectoryDatasetTest = strDirectoryBestResults + "/TestResults";
        fileBestResults = strDirectoryBestResults + "/results.txt";
        fileBestResultsAUPRCClasses = strDirectoryBestResults + "/resultsAUPRCClasses.txt";
        fileBestNeuralNetwork = strDirectoryBestResults + "/bestNeuralNetModel.ser";
        fileBestBiasWeights = strDirectoryBestResults + "/bestBiasWeights.ser";
        fileMeanSquareErrorsTraining = strDirectoryBestResults + "/meanSquareErrorsTraining.txt";
        filePredictionRealNumbers = strDirectoryDatasetTest + "/predictions.txt";
        //strDirectoryPRCurvesTest = "Results/Multilabel/" + dirAlg + "/PR_Curves/" + dirErr + "/Run" + nRun + "/" + nameDatasetTest;
        fileInterpolations = strDirectoryDatasetTest + "/interpolations.txt";
        fileInterpolationsClasses = strDirectoryDatasetTest + "/interpolationsClasses.txt";
        fileFmeasureLevels = strDirectoryDatasetTest + "/fmeasureLevels.txt";
         
    }
    
    public void setPaths(String dirAlg, String dirErr,
            String nameDatasetTest, int numRun, double threshold) {

        filePath = "Results/" + dirAlg + "/" + dirErr + "/Run" + numRun + "/" + nameDatasetTest + "/TestResults/" + threshold;
        //String filePathD = "Results/" + dirAlg + "/" + dirErr + "/Run" + numRun + "/" + nameDatasetTest + "/" + numberEpochs + "/";
        //String filePredictions = filePath + "/" + "predictions.txt";
        //String filePredictionsD = filePathD + "/" + "predictionsD.txt";
        fileResults = filePath + "/results.txt";
        fileResultsClasses = filePath + "/resultsClasses.txt";
        filePredictions = filePath + "/predictions.txt";

    }
    public void setPaths(String dirAlg, String dirErr) {

        strDirectoryAlgorithm = "Results/" + dirAlg + "/" + dirErr;

    }

    public String getFilePath() {
        return filePath;
    }

    public String getFileMeanSquareErrorsTraining() {
        return fileMeanSquareErrorsTraining;
    }

    public String getFilePredictions() {
        return filePredictions;
    }

    public String getFileResults() {
        return fileResults;
    }

    public String getStrDirectoryDatasetTest() {
        return strDirectoryDatasetTest;
    }

    public String getStrDirectoryDatasetValid() {
        return strDirectoryDatasetValid;
    }

    public String getFileOutput() {
        return fileOutput;
    }

    public String getStrDirectoryPRCurvesTest() {
        return strDirectoryPRCurvesTest;
    }

    public String getStrDirectoryPRCurvesValid() {
        return strDirectoryPRCurvesValid;
    }

    public String getFileConfusionMatrices() {
        return fileConfusionMatrices;
    }

    public String getFileBestResults() {
        return fileBestResults;
    }

    public String getStrDirectoryBestResults() {
        return strDirectoryBestResults;
    }

    public String getFileBestNeuralNetwork() {
        return fileBestNeuralNetwork;
    }

    public String getStrDirectoryAlgorithm() {
        return strDirectoryAlgorithm;
    }

    public String getFilePredictionRealNumbers() {
        return filePredictionRealNumbers;
    }

    public String getFileInterpolations() {
        return fileInterpolations;
    }
    
    public String getFileInterpolationsClasses() {
        return fileInterpolationsClasses;
    }

    public String getFileBestBiasWeights() {
        return fileBestBiasWeights;
    }
    
    public String getFileResultsClasses() {
        return fileResultsClasses;
    }
    
    public String getFileBestResultsAUPRCClasses() {
        return fileBestResultsAUPRCClasses;
    }

    public String getFileFmeasureLevels() {
        return fileFmeasureLevels;
    }
}
