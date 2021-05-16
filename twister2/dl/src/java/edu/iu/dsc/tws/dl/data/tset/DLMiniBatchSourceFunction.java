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

public class DLMiniBatchSourceFunction extends BaseSourceFunc<MiniBatch> {

  private DataSource<String, FileInputSplit<String>> dataSource;
  private InputSplit<String> dataSplit;
  private TSetContext context;
  private Pattern pattern;

  private String dataFilePath;
  private int dataSize;
  private int parallel;
  private int batchSize;
  private int count = 0;
  private boolean isFloatType = false;

  public DLMiniBatchSourceFunction(String filePath, int batchsize, int datasize, int parallelism,
                                   boolean isFloat) {
    this.dataFilePath = filePath;
    this.batchSize = batchsize;
    this.dataSize = datasize;
    this.parallel = parallelism;
    this.pattern = Pattern.compile(CSVInputSplit.DEFAULT_FIELD_DELIMITER);
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
      DenseTensor batch = new DenseTensor(false);
      int i = 0;
      int rowSize = 0;
      List<String> dataString = new ArrayList<String>();
      while (i < batchSize && !dataSplit.reachedEnd()) {
        String[] entries = pattern.split(dataSplit.nextRecord(null));
        dataString.addAll(Arrays.asList(entries));
        rowSize = entries.length;
        i++;
      }
      if (this.isFloatType) {
        float[] data = new float[dataString.size()];
        for (int j = 0; j < dataString.size(); j++) {
          data[j] = Float.parseFloat(dataString.get(j));
        }

        int[] sizes = new int[]{i, 1, rowSize};
        batch = new DenseTensor(true);
        batch.set(new ArrayFloatStorage(data), 1, sizes, null);
        return new ArrayTensorMiniBatch(batch, batch);
      } else {
        double[] data = new double[dataString.size()];
        for (int j = 0; j < dataString.size(); j++) {
          data[j] = Double.parseDouble(dataString.get(j));
        }

        int[] sizes = new int[]{i, 1, rowSize};
        batch = new DenseTensor(false);
        batch.set(new ArrayDoubleStorage(data), 1, sizes, null);
        return new ArrayTensorMiniBatch(batch, batch);
      }
    } catch (IOException e) {
      throw new RuntimeException("Unable read data split", e);
    }
  }
}
