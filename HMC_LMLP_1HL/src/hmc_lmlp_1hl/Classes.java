/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package hmc_lmlp_1hl;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 *
 * @author cerri
 */
public class Classes {

    private static ArrayList<ArrayList<String>> DAGrelationships;
    private static String[] classes;
    private static ArrayList<ArrayList<String[]>> allPathsDAGClasses;

    /* ===========================================================
     * Set a vector with all classes of the DAG structure
     * =========================================================== */
    public static void setDAGclasses(String lineClasses) {

        DAGrelationships = new ArrayList<ArrayList<String>>();

        ArrayList<String> classesAux = new ArrayList<String>();

        String[] vetLine = lineClasses.split("hierarchical");
        String[] classesAux2 = vetLine[1].split(",");

        //These arrays will store the class relationships
        //parents.get(pos) is the superclass of children.get(pos)
        ArrayList<String> parents = new ArrayList<String>();
        ArrayList<String> children = new ArrayList<String>();

        classesAux.add(classesAux2[0].split("/")[1].trim());

        parents.add(classesAux2[0].split("/")[0].trim());
        children.add(classesAux2[0].split("/")[1].trim());

        for (int i = 1; i < classesAux2.length; i++) {
            String[] classesAux3 = classesAux2[i].split("/");

            parents.add(classesAux3[0].trim());
            children.add(classesAux3[1].trim());

            if (!classesAux.contains(classesAux3[0]) && !"root".equals(classesAux3[0])) {
                classesAux.add(classesAux3[0].trim());
            }

            if (!classesAux.contains(classesAux3[1])) {
                classesAux.add(classesAux3[1].trim());
            }
        }

        //Vector with all classes
        classes = new String[classesAux.size()];
        for (int i = 0; i < classesAux.size(); i++) {
            classes[i] = classesAux.get(i);
        }

        //Class relationships
        DAGrelationships.add(parents);
        DAGrelationships.add(children);

        //System.out.println();
    }

    /*============================================================================
     * Get the file which contains the DAG relationships
     *============================================================================*/
    public static void setClassesPaths(String pathDatasets, String pathConfigFile) {

        String fileDAGpaths = "";

        try {
            File configFile = new File(pathConfigFile);
            FileReader reader = new FileReader(configFile);
            BufferedReader buffReader = new BufferedReader(reader);

            String regExp = "DAG paths = ";
            Pattern pattern = Pattern.compile(regExp);

            String line = null;
            while ((line = buffReader.readLine()) != null) {
                Matcher m = pattern.matcher(line);
                if (m.find()) {
                    String[] vectorLine = line.split(" = ");
                    fileDAGpaths = vectorLine[1];
                    break;
                }
            }

            buffReader.close();
            reader.close();

            File DAGFile = new File(pathDatasets + fileDAGpaths);
            reader = new FileReader(DAGFile);
            buffReader = new BufferedReader(reader);

            line = null;
            allPathsDAGClasses = new ArrayList<ArrayList<String[]>>();
            
            while ((line = buffReader.readLine()) != null) {
                
                ArrayList<String[]> paths = new ArrayList<String[]>();
                line = line.trim();
                String[] vectorClasses = line.split("@");
                for(int i=0; i<vectorClasses.length; i++){
                    paths.add(vectorClasses[i].split("/"));
                }
                
                allPathsDAGClasses.add(paths);               
            }

            buffReader.close();
            reader.close();

        } catch (IOException ioe) {
            ioe.printStackTrace();
        }

    }

    public static ArrayList<ArrayList<String>> getDAGrelationships() {
        return DAGrelationships;
    }

    public static String[] getClasses() {
        return classes;
    }
    
    public static ArrayList<ArrayList<String[]>> getAllPathsDAGClasses() {
        return allPathsDAGClasses;
    }
}
