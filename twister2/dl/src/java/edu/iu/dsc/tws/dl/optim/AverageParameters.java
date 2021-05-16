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

import edu.iu.dsc.tws.api.tset.fn.BaseMapFunc;
import edu.iu.dsc.tws.dl.data.TensorNumeric;
import edu.iu.dsc.tws.dl.utils.pair.DoubleDoubleArrayPair;
import edu.iu.dsc.tws.dl.utils.pair.FloatFloatArrayPair;
import edu.iu.dsc.tws.dl.utils.pair.PrimitiveArrayPair;

public class AverageParameters extends BaseMapFunc<PrimitiveArrayPair, PrimitiveArrayPair> {

  @Override
  public PrimitiveArrayPair map(PrimitiveArrayPair input) {
    long startTime = System.nanoTime();
    int parallelism = this.getTSetContext().getParallelism();


    if (input instanceof DoubleDoubleArrayPair) {
      DoubleDoubleArrayPair tempInput = (DoubleDoubleArrayPair) input;
      tempInput.setValue0(tempInput.getValue0() / parallelism);
      double[] data = tempInput.getValue1();
      TensorNumeric.scal(data.length, 1.0 / parallelism, data, 0, 1);
    } else {
      FloatFloatArrayPair tempInput = (FloatFloatArrayPair) input;
      tempInput.setValue0(tempInput.getValue0() / parallelism);
      float[] data = tempInput.getValue1();
      TensorNumeric.scal(data.length, 1.0f / parallelism, data, 0, 1);
    }
    if (this.getTSetContext().getIndex() == 0) {
      System.out.println("" + (System.nanoTime() - startTime) / 1e6);
    }
    return input;
  }
}
