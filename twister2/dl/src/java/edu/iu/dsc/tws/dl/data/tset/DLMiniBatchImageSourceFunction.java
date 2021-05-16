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
package edu.iu.dsc.tws.dl.data.tset;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.regex.Pattern;

import edu.iu.dsc.tws.api.config.Config;
import edu.iu.dsc.tws.api.data.Path;
import edu.iu.dsc.tws.api.tset.TSetContext;
import edu.iu.dsc.tws.api.tset.fn.BaseSourceFunc;
import edu.iu.dsc.tws.data.api.formatters.LocalCSVInputPartitioner;
import edu.iu.dsc.tws.data.api.splits.CSVInputSplit;
import edu.iu.dsc.tws.data.api.splits.FileInputSplit;
import edu.iu.dsc.tws.data.fs.io.InputSplit;
import edu.iu.dsc.tws.dataset.DataSource;
import edu.iu.dsc.tws.dl.data.MiniBatch;
import edu.iu.dsc.tws.dl.data.minibatch.ArrayTensorMiniBatch;
import edu.iu.dsc.tws.dl.data.storage.ArrayDoubleStorage;
import edu.iu.dsc.tws.dl.data.storage.ArrayFloatStorage;
import edu.iu.dsc.tws.dl.data.tensor.DenseTensor;

public class DLMiniBatchImageSourceFunction extends BaseSourceFunc<MiniBatch> {

  private DataSource<String, FileInputSplit<String>> dataSource;
  private InputSplit<String> dataSplit;
  private TSetContext context;
  private Pattern pattern;

  private String dataFilePath;
  private int dataSize;
  private int parallel;
  private int batchSize;
  private int count = 0;
  private int nPlanes = 1;
  private int w = 1;
  private int h = 1;
  private boolean isFloatType;


  public DLMiniBatchImageSourceFunction(String filePath, int batchsize, int datasize,
                                        int parallelism, int planes, int width, int height,
                                        boolean isFloat) {
    this.dataFilePath = filePath;
    this.batchSize = batchsize;
    this.dataSize = datasize;
    this.parallel = parallelism;
    this.pattern = Pattern.compile(CSVInputSplit.DEFAULT_FIELD_DELIMITER);
    this.nPlanes = planes;
    this.w = width;
    this.h = height;
    this.isFloatType = isFloat;
  }

  @Override
  public void prepare(TSetContext ctx) {
    super.prepare(ctx);
    this.context = ctx;
    count = count + 1;
    Config cfg = ctx.getConfig();
    this.dataSource = new DataSource(cfg, new LocalCSVInputPartitioner(
        new Path(dataFilePath), context.getParallelism(), dataSize, cfg), parallel);
    this.dataSplit = this.dataSource.getNextSplit(context.getIndex());
  }

  @Override
  public boolean hasNext() {
    try {
      if (dataSplit == null || dataSplit.reachedEnd()) {
        dataSplit = dataSource.getNextSplit(getTSetContext().getIndex());
      }
      return dataSplit != null && !dataSplit.reachedEnd();
    } catch (IOException e) {
      throw new RuntimeException("Unable read data split", e);
    }
  }

  @Override
  public MiniBatch next() {
    try {
      if (this.isFloatType) {
        DenseTensor features;
        DenseTensor lables;
        int realBatchSize = 0;
        int rowSize = 0;
        List<String> dataString = new ArrayList<String>();
        while (realBatchSize < batchSize && !dataSplit.reachedEnd()) {
          String[] entries = pattern.split(dataSplit.nextRecord(null));
          dataString.addAll(Arrays.asList(entries));
          rowSize = entries.length;
          realBatchSize++;
        }
        float[] data = new float[(rowSize - 1) * realBatchSize];
        float[] label = new float[realBatchSize];
        int currBatch = 0;
        for (int j = 0; j < dataString.size(); j++) {
          if (j % rowSize == 0) {
            label[currBatch] = Float.parseFloat(dataString.get(j)) + 1;
            currBatch++;
          } else {
            data[j - currBatch] = Float.parseFloat(dataString.get(j));
          }
        }

        int[] sizesD = new int[]{realBatchSize, nPlanes, w, h};
        int[] sizesL = new int[]{realBatchSize, 1};
        features = new DenseTensor(true);
        lables = new DenseTensor(true);
        features.set(new ArrayFloatStorage(data), 1, sizesD, null);
        lables.set(new ArrayFloatStorage(label), 1, sizesL, null);
        return new ArrayTensorMiniBatch(features, lables);
      } else {
        DenseTensor features;
        DenseTensor lables;
        int realBatchSize = 0;
        int rowSize = 0;
        List<String> dataString = new ArrayList<String>();
        while (realBatchSize < batchSize && !dataSplit.reachedEnd()) {
          String[] entries = pattern.split(dataSplit.nextRecord(null));
          dataString.addAll(Arrays.asList(entries));
          rowSize = entries.length;
          realBatchSize++;
        }
        double[] data = new double[(rowSize - 1) * realBatchSize];
        double[] label = new double[realBatchSize];
        int currBatch = 0;
        for (int j = 0; j < dataString.size(); j++) {
          if (j % rowSize == 0) {
            label[currBatch] = Double.parseDouble(dataString.get(j)) + 1;
            currBatch++;
          } else {
            data[j - currBatch] = Double.parseDouble(dataString.get(j));
          }
        }

        int[] sizesD = new int[]{realBatchSize, nPlanes, w, h};
        int[] sizesL = new int[]{realBatchSize, 1};
        features = new DenseTensor(false);
        lables = new DenseTensor(false);
        features.set(new ArrayDoubleStorage(data), 1, sizesD, null);
        lables.set(new ArrayDoubleStorage(label), 1, sizesL, null);
        return new ArrayTensorMiniBatch(features, lables);
      }
    } catch (IOException e) {
      throw new RuntimeException("Unable read data split", e);
    }
  }
}
