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
package edu.iu.dsc.tws.data.utils;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.logging.Logger;

import edu.iu.dsc.tws.api.config.Config;
import edu.iu.dsc.tws.api.data.FSDataInputStream;
import edu.iu.dsc.tws.api.data.FileStatus;
import edu.iu.dsc.tws.api.data.FileSystem;
import edu.iu.dsc.tws.api.data.Path;

/**
 * This class acts as an interface for reading the input datapoints and centroid values from
 * the local file system or from the distributed file system (HDFS).
 */
public class DataFileReader {

  private static final Logger LOG = Logger.getLogger(DataFileReader.class.getName());

  private final Config config;
  private final String fileSystem;

  private volatile FSDataInputStream fdis;

  public DataFileReader(Config cfg, String fileSys) {
    this.config = cfg;
    this.fileSystem = fileSys;
  }

  /**
   * It reads the datapoints from the corresponding file and store the data in a two-dimensional
   * array for the later processing. The size of the two-dimensional array should be equal to the
   * number of clusters and the dimension considered for the clustering process.
   */
  public double[][] readData(Path path, int dimension, int datasize) {
    double[][] datapoints = new double[datasize][dimension];
    final FileStatus pathFile;
    try {
      final FileSystem fs = FileSystemUtils.get(path, config);
      if (DataContext.TWISTER2_HDFS_FILESYSTEM.equals(fileSystem)) {
        pathFile = fs.getFileStatus(path);
        this.fdis = fs.open(pathFile.getPath());
      } else {
        for (FileStatus file : fs.listFiles(path)) {
          this.fdis = fs.open(file.getPath());
        }
      }
      BufferedReader bufferedReader = new BufferedReader(new InputStreamReader(this.fdis));
      String line;
      int value = 0;
      while ((line = bufferedReader.readLine()) != null) {
        String[] data = line.split(",");
        for (int i = 0; i < data.length - 1; i++) {
          datapoints[value][i] = Double.parseDouble(data[i].trim());
          datapoints[value][i + 1] = Double.parseDouble(data[i + 1].trim());
        }
        value++;
      }
      if (bufferedReader != null) {
        bufferedReader.close();
      }
    } catch (IOException ioe) {
      throw new RuntimeException("IO Exception Occured");
    }
    return datapoints;
  }
}

