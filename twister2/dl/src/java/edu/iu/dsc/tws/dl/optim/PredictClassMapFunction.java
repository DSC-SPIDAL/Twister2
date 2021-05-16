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
import edu.iu.dsc.tws.dl.data.Activity;
import edu.iu.dsc.tws.dl.data.Tensor;
import edu.iu.dsc.tws.dl.data.minibatch.ArrayTensorMiniBatch;
import edu.iu.dsc.tws.dl.module.AbstractModule;
import edu.iu.dsc.tws.dl.utils.pair.TensorPair;

public class PredictClassMapFunction<A extends Tensor, T> extends BaseMapFunc<T, int[]> {

  private AbstractModule<A> modal;

  public PredictClassMapFunction(AbstractModule predictionModal) {
    this.modal = predictionModal;
  }

  @Override
  public int[] map(T input) {

    ArrayTensorMiniBatch miniBatch = (ArrayTensorMiniBatch) input;
    modal.evaluate();
    Activity data = miniBatch.getInput();
    A outputProbs = modal.forward((A) data);
    TensorPair output = outputProbs.max(2);
    A classes = (A) output.getValue1();
    double[] tempres = classes.storage().toDoubleArray();
    int[] results = new int[classes.size(1)];
    for (int i = 0; i < results.length; i++) {
      results[i] = (int) tempres[i];
    }
    return results;
  }
}
