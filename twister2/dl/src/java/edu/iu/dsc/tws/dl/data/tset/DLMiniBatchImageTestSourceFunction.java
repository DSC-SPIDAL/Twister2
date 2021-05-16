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

import java.util.concurrent.ThreadLocalRandom;

import edu.iu.dsc.tws.api.tset.TSetContext;
import edu.iu.dsc.tws.api.tset.fn.BaseSourceFunc;
import edu.iu.dsc.tws.dl.data.MiniBatch;
import edu.iu.dsc.tws.dl.data.minibatch.ArrayTensorMiniBatch;
import edu.iu.dsc.tws.dl.data.storage.ArrayDoubleStorage;
import edu.iu.dsc.tws.dl.data.storage.ArrayFloatStorage;
import edu.iu.dsc.tws.dl.data.tensor.DenseTensor;

public class DLMiniBatchImageTestSourceFunction extends BaseSourceFunc<MiniBatch> {

  private int numSplits;
  private int currentSplit;
  private int batchSize;
  private boolean isFloatType;
  private int inputSize;
  private int nPlanes = 1;
  private int w = 1;
  private int h = 1;

  public DLMiniBatchImageTestSourceFunction(int batchsize, int datasize,
                                            int parallelism, int planes, int width, int height,
                                            boolean isFloat, int inSize) {
    this.batchSize = batchsize;
    this.isFloatType = isFloat;
    this.numSplits = datasize / (parallelism * batchsize);
    this.currentSplit = 1;
    this.inputSize = inSize;
    this.nPlanes = planes;
    this.w = width;
    this.h = height;
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
    this.currentSplit++;
    if (!this.isFloatType) {
      DenseTensor features;
      DenseTensor lables;
      double[] data = new double[this.batchSize * this.inputSize];
      double[] label = new double[this.batchSize];
      for (int j = 0; j < data.length; j++) {
        data[j] = Math.random();
      }

      for (int i = 0; i < label.length; i++) {
        label[i] = ThreadLocalRandom.current().nextInt(1, 9 + 1);
      }

      int[] sizesD = new int[]{this.batchSize, nPlanes, w, h};
      int[] sizesL = new int[]{this.batchSize, 1};
      features = new DenseTensor(false);
      lables = new DenseTensor(false);
      features.set(new ArrayDoubleStorage(data), 1, sizesD, null);
      lables.set(new ArrayDoubleStorage(label), 1, sizesL, null);
      return new ArrayTensorMiniBatch(features, lables);
    } else {
      DenseTensor features;
      DenseTensor lables;
      float[] data = new float[this.batchSize * this.inputSize];
      float[] label = new float[this.batchSize];
      for (int j = 0; j < data.length; j++) {
        data[j] = (float) Math.random();
      }

      for (int i = 0; i < label.length; i++) {
        label[i] = ThreadLocalRandom.current().nextInt(1, 9 + 1);
      }

      int[] sizesD = new int[]{this.batchSize, nPlanes, w, h};
      int[] sizesL = new int[]{this.batchSize, 1};
      features = new DenseTensor(true);
      lables = new DenseTensor(true);
      features.set(new ArrayFloatStorage(data), 1, sizesD, null);
      lables.set(new ArrayFloatStorage(label), 1, sizesL, null);
      return new ArrayTensorMiniBatch(features, lables);
    }
  }
}
