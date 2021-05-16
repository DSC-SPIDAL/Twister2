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
package edu.iu.dsc.tws.dl.optim;

import edu.iu.dsc.tws.api.tset.fn.ReduceFunc;
import edu.iu.dsc.tws.dl.data.TensorNumeric;
import edu.iu.dsc.tws.dl.utils.pair.DoubleDoubleArrayPair;
import edu.iu.dsc.tws.dl.utils.pair.FloatFloatArrayPair;
import edu.iu.dsc.tws.dl.utils.pair.PrimitiveArrayPair;

public class TrainReduceFunction implements ReduceFunc<PrimitiveArrayPair> {

  private PrimitiveArrayPair data;
  private int index;
  private boolean isFloat;

  public TrainReduceFunction(PrimitiveArrayPair result, int rank, boolean isf) {
    this.data = result;
    this.index = rank;
    this.isFloat = isf;
  }

  @Override
  public PrimitiveArrayPair reduce(PrimitiveArrayPair t1, PrimitiveArrayPair t2) {
    long startTime = System.nanoTime();
    if (!this.isFloat) {
      DoubleDoubleArrayPair tempdata = (DoubleDoubleArrayPair) this.data;
      DoubleDoubleArrayPair temp1 = (DoubleDoubleArrayPair) t1;
      DoubleDoubleArrayPair temp2 = (DoubleDoubleArrayPair) t2;
      tempdata.setValue0(temp1.getValue0() + temp2.getValue0());
      double[] grad = tempdata.getValue1();
      TensorNumeric.vAdd(grad.length, temp1.getValue1(), 0, temp2.getValue1(), 0, grad, 0);
    } else {
      FloatFloatArrayPair tempdata = (FloatFloatArrayPair) this.data;
      FloatFloatArrayPair temp1 = (FloatFloatArrayPair) t1;
      FloatFloatArrayPair temp2 = (FloatFloatArrayPair) t2;
      tempdata.setValue0(temp1.getValue0() + temp2.getValue0());
      float[] grad = tempdata.getValue1();
      TensorNumeric.vAdd(grad.length, temp1.getValue1(), 0, temp2.getValue1(), 0, grad, 0);
    }

    if (this.index == 0) {
      System.out.println("" + (System.nanoTime() - startTime) / 1e6);
    }
    return this.data;
  }
}
