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
import java.util.Arrays;
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
import edu.iu.dsc.tws.dl.data.Sample;
import edu.iu.dsc.tws.dl.data.sample.TensorSample;
import edu.iu.dsc.tws.dl.data.tensor.DenseTensor;

public class DLBasicSourceFunction extends BaseSourceFunc<Sample> {

  private DataSource<String, FileInputSplit<String>> dataSource;
  private InputSplit<String> dataSplit;
  private TSetContext context;
  private Pattern pattern;

  private String dataFilePath;
  private int dataSize;
  private int parallel;

  public DLBasicSourceFunction(String filePath, int datasize, int parallelism) {
    this.dataFilePath = filePath;
    this.dataSize = datasize;
    this.parallel = parallelism;
    this.pattern = Pattern.compile(CSVInputSplit.DEFAULT_FIELD_DELIMITER);
  }

  @Override
  public void prepare(TSetContext ctx) {
    super.prepare(ctx);
    this.context = ctx;
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
  public Sample next() {
    try {
      String[] entries = pattern.split(dataSplit.nextRecord(null));
      double[] data = Arrays.stream(entries).mapToDouble(Double::parseDouble)
          .toArray();
      return new TensorSample(new DenseTensor(data));
    } catch (IOException e) {
      throw new RuntimeException("Unable read data split", e);
    }
  }
}
