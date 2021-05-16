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

import edu.iu.dsc.tws.api.tset.TSetContext;
import edu.iu.dsc.tws.api.tset.fn.BaseSourceFunc;
import edu.iu.dsc.tws.dl.data.MiniBatch;
import edu.iu.dsc.tws.dl.data.minibatch.ArrayTensorMiniBatch;
import edu.iu.dsc.tws.dl.data.storage.ArrayDoubleStorage;
import edu.iu.dsc.tws.dl.data.storage.ArrayFloatStorage;
import edu.iu.dsc.tws.dl.data.tensor.DenseTensor;

public class DLMiniBatchTestSourceFunction extends BaseSourceFunc<MiniBatch> {

  private int batchSize;
  private int numSplits;
  private int currentSplit;
  private boolean isFloatType;
  private int inputSize;

  public DLMiniBatchTestSourceFunction(int batchsize, int datasize, int parallelism,
                                       boolean isFloat, int inSize) {
    this.batchSize = batchsize;
    this.isFloatType = isFloat;
    this.numSplits = datasize / (parallelism * batchsize);
    this.currentSplit = 1;
    this.inputSize = inSize;
  }

  @Override
  public void prepare(TSetContext ctx) {
    super.prepare(ctx);
  }

  @Override
  public boolean hasNext() {
    return currentSplit <= numSplits;
  }

  @Override
  public MiniBatch next() {
    DenseTensor batch = new DenseTensor(false);
    this.currentSplit++;
    if (this.isFloatType) {
      float[] data = new float[this.batchSize * this.inputSize];
      for (int j = 0; j < data.length; j++) {
        data[j] = (float) Math.random();
      }

      int[] sizes = new int[]{this.batchSize, 1, inputSize};
      batch = new DenseTensor(true);
      batch.set(new ArrayFloatStorage(data), 1, sizes, null);
      return new ArrayTensorMiniBatch(batch, batch);
    } else {
      double[] data = new double[this.batchSize * this.inputSize];
      for (int j = 0; j < data.length; j++) {
        data[j] = Math.random();
      }

      int[] sizes = new int[]{this.batchSize, 1, inputSize};
      batch = new DenseTensor(false);
      batch.set(new ArrayDoubleStorage(data), 1, sizes, null);
      return new ArrayTensorMiniBatch(batch, batch);
    }
  }
}
