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
import edu.iu.dsc.tws.dl.criterion.AbstractCriterion;
import edu.iu.dsc.tws.dl.data.Activity;
import edu.iu.dsc.tws.dl.data.Tensor;
import edu.iu.dsc.tws.dl.data.minibatch.ArrayTensorMiniBatch;
import edu.iu.dsc.tws.dl.module.AbstractModule;
import edu.iu.dsc.tws.dl.utils.pair.DoubleDoubleArrayPair;
import edu.iu.dsc.tws.dl.utils.pair.FloatFloatArrayPair;
import edu.iu.dsc.tws.dl.utils.pair.PrimitiveArrayPair;

public class TrainMapFunction<A extends Tensor, T> extends BaseMapFunc<T, PrimitiveArrayPair> {

  private AbstractModule<A> modal;
  private AbstractCriterion criterion;

  public TrainMapFunction(AbstractCriterion localCriterion) {
    this.criterion = localCriterion;
  }

  @Override
  public PrimitiveArrayPair map(T data) {
    long startTime = System.nanoTime();

    if (modal == null) {
      modal = (AbstractModule) getTSetContext()
          .getInput("modal").getConsumer().next();
    } else {
      AbstractModule temp = (AbstractModule) getTSetContext()
          .getInput("modal").getConsumer().next();
      modal.setModules(temp.getModules());
    }

    ArrayTensorMiniBatch miniBatch = (ArrayTensorMiniBatch) data;
    modal.zeroGradParameters();
    modal.training();
    Activity input = miniBatch.getInput();
    Activity target = miniBatch.getTarget();
    long startTimef = System.nanoTime();
    A output = modal.forward((A) input);
    PrimitiveArrayPair result;
    if (modal.isFloat()) {
      float loss = criterion.forwardf(output, target);
     // System.out.println("Forward Time : " + (System.nanoTime() - startTimef) / 1e6);
      startTimef = System.nanoTime();
      Activity errors = criterion.backward(output, target);
      modal.backward((A) input, (A) errors);
      result = new FloatFloatArrayPair(loss,
          modal.getParameters().getValue1().storage().toFloatArray());
     // System.out.println("Backward Time : " + (System.nanoTime() - startTimef) / 1e6);
      if (this.getTSetContext().getIndex() == 0) {
//        System.out.println("Iteration for " + ((Tensor) input).size()[0]
//            + "records time : " + (System.nanoTime() - startTime) / 1e6);
        System.out.println("" + (System.nanoTime() - startTime) / 1e6);
      }
    } else {
      double loss = criterion.forward(output, target);
      Activity errors = criterion.backward(output, target);
      modal.backward((A) input, (A) errors);
      result = new DoubleDoubleArrayPair(loss,
          modal.getParameters().getValue1().storage().toDoubleArray());
      if (this.getTSetContext().getIndex() == 0) {
//        System.out.println("Iteration time : " + (System.nanoTime() - startTime) / 1e6);
        System.out.println("" + (System.nanoTime() - startTime) / 1e6);
      }
    }

    return result;
  }
}
