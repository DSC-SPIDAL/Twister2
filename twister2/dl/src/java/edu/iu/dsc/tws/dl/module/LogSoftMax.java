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
package edu.iu.dsc.tws.dl.module;

import edu.iu.dsc.tws.dl.data.TensorNumeric;
import edu.iu.dsc.tws.dl.data.tensor.DenseTensor;
import edu.iu.dsc.tws.dl.graph.TensorModule;
import edu.iu.dsc.tws.dl.utils.ErrorConstants;
import edu.iu.dsc.tws.dl.utils.Shape;
import edu.iu.dsc.tws.dl.utils.Util;
import edu.iu.dsc.tws.dl.utils.pair.TensorArrayPair;

/**
 * The [[LogSoftMax]] module applies a LogSoftMax transformation to the input data
 * which is defined as:
 * f_i(x) = log(1 / a exp(x_i))
 * where a = sum_j[exp(x_j)]
 * <p>
 * The input given in `forward(input)` must be either
 * a vector (1D tensor) or matrix (2D tensor).
 */
public class LogSoftMax extends TensorModule<DenseTensor> {

  private DenseTensor ones = new DenseTensor(false);
  private DenseTensor buffer = new DenseTensor(false);

  public LogSoftMax() {
    this.output = new DenseTensor(this.isFloat);
    this.gradInput = new DenseTensor(this.isFloat);
  }

  @Override
  public void toFloat() {
    super.toFloat();
    ones = new DenseTensor(true);
    buffer = new DenseTensor(true);
    this.output = new DenseTensor(this.isFloat);
    this.gradInput = new DenseTensor(this.isFloat);
  }

  @Override
  public DenseTensor updateOutput(DenseTensor input) {
    Util.require(input.dim() == 1 || input.dim() == 2,
        "LogSoftMax: " + ErrorConstants.constrainInputAsVectorOrBatch
            + "input dim ${input.dim()}");
    ((DenseTensor) output).resizeAs(input).copy(input);

    int nframe;
    int dim;
    if (input.nDimension() == 1) {
      nframe = 1;
      dim = input.size(1);
    } else {
      nframe = input.size(1);
      dim = input.size(2);
    }

    //TODO parallelize
    if (nframe == 1) {
      updateOutputFrame(input, (DenseTensor) output);
    } else {

      for (int t = 1; t <= nframe; t++) {
        updateOutputFrame((DenseTensor) input.select(1, t),
            (DenseTensor) ((DenseTensor) output).select(1, t));
      }
    }

    return (DenseTensor) output;
  }

  private void updateOutputFrame(DenseTensor in, DenseTensor out) {

    if (this.isFloat) {
      if (ones.nElement() < in.nElement()) {
        ones.resizeAs(in).fill(1.0f);
      }
      if (buffer.nElement() != out.nElement()) {
        buffer.resizeAs(out);
      }
      // use exp(in - maxInput) to avoid Infinity error
      float maxInput = in.maxf();

      buffer.fill(TensorNumeric.negative(maxInput));
      buffer.add(in);
      buffer.exp();
      float logSum = TensorNumeric.plus(maxInput, TensorNumeric.log(buffer.dotf(ones)));

      out.add(TensorNumeric.negative(logSum));
    } else {
      if (ones.nElement() < in.nElement()) {
        ones.resizeAs(in).fill(1.0);
      }
      if (buffer.nElement() != out.nElement()) {
        buffer.resizeAs(out);
      }
      // use exp(in - maxInput) to avoid Infinity error
      double maxInput = in.max();

      buffer.fill(TensorNumeric.negative(maxInput));
      buffer.add(in);
      buffer.exp();
      double logSum = TensorNumeric.plus(maxInput, TensorNumeric.log(buffer.dot(ones)));

      out.add(TensorNumeric.negative(logSum));
    }
  }


  @Override
  public DenseTensor updateGradInput(DenseTensor input, DenseTensor gradOutput) {
    Util.require(((DenseTensor) output).nDimension() == 1
        || ((DenseTensor) output).nDimension() == 2, "vector or matrix expected");
    Util.require(gradOutput.dim() == input.dim(),
        "LogSoftMax: input and gradOutput shapes do not "
            + "match, input_dim: " + input.dim() + ", gradOutput_dim: " + gradOutput.dim());

    ((DenseTensor) gradInput).resizeAs(input).copy(gradOutput);
    int nframe;
    int dim;
    if (input.nDimension() == 1) {
      nframe = 1;
      dim = ((DenseTensor) output).size(1);
    } else {
      nframe = ((DenseTensor) output).size(1);
      dim = ((DenseTensor) output).size(2);
    }
    if (nframe == 1) {
      updateGradInputFrame((DenseTensor) output, (DenseTensor) gradInput);
    } else {
      for (int t = 1; t <= nframe; t++) {
        updateGradInputFrame((DenseTensor) ((DenseTensor) output).select(1, t),
            (DenseTensor) ((DenseTensor) gradInput).select(1, t));
      }
    }
    return (DenseTensor) gradInput;
  }

  private void updateGradInputFrame(DenseTensor out, DenseTensor gradOut) {
    if (this.isFloat) {
      buffer.exp(out);
      float outSum = gradOut.dotf(ones);
      gradOut.add(TensorNumeric.negative(outSum), buffer);
    } else {
      buffer.exp(out);
      double outSum = gradOut.dot(ones);
      gradOut.add(TensorNumeric.negative(outSum), buffer);
    }
  }

  @Override
  public TensorArrayPair parameters() {
    return null;
  }

  @Override
  public AbstractModule clearState() {
    super.clearState();
    ones.set();
    buffer.set();
    return this;
  }

  @Override
  public Shape computeOutputShape(Shape inputShape) {
    return inputShape;
  }

  @Override
  public void reset() {

  }
}
