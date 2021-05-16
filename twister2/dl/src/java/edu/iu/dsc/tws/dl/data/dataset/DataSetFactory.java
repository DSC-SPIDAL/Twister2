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
package edu.iu.dsc.tws.dl.data.dataset;

import edu.iu.dsc.tws.api.dataset.DataObject;
import edu.iu.dsc.tws.api.dataset.DataPartition;
import edu.iu.dsc.tws.dataset.DataObjectImpl;
import edu.iu.dsc.tws.dataset.partition.EntityPartition;
import edu.iu.dsc.tws.dl.data.MiniBatch;
import edu.iu.dsc.tws.dl.data.Sample;
import edu.iu.dsc.tws.dl.data.tset.DLBasicSourceFunction;
import edu.iu.dsc.tws.dl.data.tset.DLMiniBatchImageSourceFunction;
import edu.iu.dsc.tws.dl.data.tset.DLMiniBatchImageTestSourceFunction;
import edu.iu.dsc.tws.dl.data.tset.DLMiniBatchSourceFunction;
import edu.iu.dsc.tws.dl.data.tset.DLMiniBatchTestSourceFunction;
import edu.iu.dsc.tws.dl.data.tset.ModalSource;
import edu.iu.dsc.tws.dl.data.tset.SingleDataSource;
import edu.iu.dsc.tws.dl.module.AbstractModule;
import edu.iu.dsc.tws.tset.env.BatchEnvironment;
import edu.iu.dsc.tws.tset.sets.batch.SourceTSet;

public final class DataSetFactory {

  private DataSetFactory() {
  }

  public static SourceTSet<MiniBatch> createImageMiniBatchDataSet(BatchEnvironment env,
                                                                  String filePath, int nPlanes,
                                                                  int width, int height,
                                                                  int batchSize, int dataSize,
                                                                  int parallelism,
                                                                  boolean isFloat) {
    return env.createSource(new DLMiniBatchImageSourceFunction(filePath,
        batchSize, dataSize, parallelism, nPlanes, width, height, isFloat), parallelism);
  }

  public static SourceTSet<MiniBatch> createImageMiniBatchTestDataSet(BatchEnvironment env,
                                                                  int nPlanes,
                                                                  int width, int height,
                                                                  int batchSize, int dataSize,
                                                                  int parallelism,
                                                                  boolean isFloat, int inSize) {
    return env.createSource(new DLMiniBatchImageTestSourceFunction(
        batchSize, dataSize, parallelism, nPlanes, width, height, isFloat, inSize), parallelism);
  }

  public static SourceTSet<MiniBatch> createMiniBatchDataSet(BatchEnvironment env, String filePath,
                                                             int batchSize, int dataSize,
                                                             int parallelism, boolean isFloat) {
    return env.createSource(new DLMiniBatchSourceFunction(filePath,
        batchSize, dataSize, parallelism, isFloat), parallelism);
  }

  public static SourceTSet<MiniBatch> createMiniBatchTestDataSet(BatchEnvironment env,
                                                                 int batchSize, int dataSize,
                                                                 int parallelism, boolean isFloat,
                                                                 int inputSize) {
    return env.createSource(new DLMiniBatchTestSourceFunction(batchSize, dataSize, parallelism,
        isFloat, inputSize), parallelism);
  }

  public static SourceTSet<Sample> createSampleDataSet(BatchEnvironment env, String filePath,
                                                       int dataSize, int parallelism) {
    return env.createSource(new DLBasicSourceFunction(filePath,
        dataSize, parallelism), parallelism);
  }

  public static <T> SourceTSet<T> createSingleDataSet(BatchEnvironment env, T data,
                                                      int parallelism) {
    return env.createSource(new SingleDataSource<T>(data), parallelism);
  }

  public static SourceTSet<AbstractModule> createModalDataSet(BatchEnvironment env,
                                                              AbstractModule data,
                                                              int parallelism) {
    return env.createSource(new ModalSource(data), parallelism);
  }

  public static <T> DataObject<T> createDataObject(BatchEnvironment env, T data) {
    DataObject<T> dataObject = new DataObjectImpl<T>("modalData", null);
    DataPartition<T> partition = new EntityPartition<T>(data);
    partition.setId(env.getWorkerID());
    dataObject.addPartition(partition);
    return dataObject;
  }
}

