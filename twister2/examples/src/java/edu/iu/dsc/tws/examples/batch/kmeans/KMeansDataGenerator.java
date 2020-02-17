//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.
package edu.iu.dsc.tws.examples.batch.kmeans;

import java.io.IOException;
import java.util.Random;
import java.util.logging.Logger;

import edu.iu.dsc.tws.api.config.Config;
import edu.iu.dsc.tws.api.data.FileSystem;
import edu.iu.dsc.tws.api.data.Path;
import edu.iu.dsc.tws.data.api.out.CSVOutputWriter;
import edu.iu.dsc.tws.data.api.out.TextOutputWriter;

/**
 * Generate a data set
 * <p>
 * 1. We can generate in each worker
 * 2. We can generate in a common location shared by workers, such as HDFS or NFS
 */
public final class KMeansDataGenerator {

  private static final Logger LOG = Logger.getLogger(KMeansDataGenerator.class.getName());

  private KMeansDataGenerator() {
  }

  /**
   * Generate a data set
   *
   * @param type       type of file, i.e. csv, text, binary
   * @param directory  the directory to generate
   * @param numOfFiles number of files to create
   * @param sizeOfFile size of each file, different types have a different meaning
   * @param sizeMargin size will be varied about this much
   */
  public static void generateData(String type, Path directory, int numOfFiles, int sizeOfFile,
                                  int sizeMargin, int dimension, Config cfg)
      throws IOException {
    if ("csv".equals(type)) {
      generateCSV(directory, numOfFiles, sizeOfFile, sizeMargin, dimension, cfg);
    } else if ("txt".equals(type)) {
      generateText(directory, numOfFiles, sizeOfFile, sizeMargin, dimension, cfg);
    } else {
      throw new IOException("Unsupported data gen type: " + type);
    }
  }

  private static void generateText(Path directory, int numOfFiles, int sizeOfFile,
                                   int sizeMargin, int dimension, Config config) {
    for (int i = 0; i < numOfFiles; i++) {
      String points = generatePoints(sizeOfFile, dimension, sizeMargin);
      TextOutputWriter textOutputWriter
          = new TextOutputWriter(FileSystem.WriteMode.OVERWRITE, directory, config);
      textOutputWriter.createOutput();
      textOutputWriter.writeRecord(points);
      textOutputWriter.close();
    }
  }

  /**
   * Generate a random csv file, we generate a csv with 10 attributes
   *
   * @param directory the path of the directory
   */
  private static void generateCSV(Path directory, int numOfFiles, int sizeOfFile,
                                  int sizeMargin, int dimension, Config config) {
    for (int i = 0; i < numOfFiles; i++) {
      String points = generatePoints(sizeOfFile, dimension, sizeMargin);
      CSVOutputWriter csvOutputWriter
          = new CSVOutputWriter(FileSystem.WriteMode.OVERWRITE, directory, config);
      csvOutputWriter.createOutput();
      csvOutputWriter.writeRecord(points);
      csvOutputWriter.close();
    }
  }

  private static String generatePoints(int numPoints, int dimension, int seedValue) {
    StringBuilder datapoints = new StringBuilder();
    Random r = new Random(seedValue);
    for (int i = 0; i < numPoints; i++) {
      StringBuilder line = new StringBuilder();
      for (int j = 0; j < dimension; j++) {
        double randomValue = r.nextDouble();
        line.append(String.format("%1$,.8f", randomValue));
        if (j < dimension - 1) {
          line.append(",").append("\t");
        }
      }
      datapoints.append(line);
      datapoints.append("\n");
    }
    return datapoints.toString();
  }
}
