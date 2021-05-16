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

import java.util.Arrays;

import edu.iu.dsc.tws.dl.data.Tensor;
import edu.iu.dsc.tws.dl.data.TensorNumeric;
import edu.iu.dsc.tws.dl.data.tensor.DenseTensor;
import edu.iu.dsc.tws.dl.graph.TensorModule;
import edu.iu.dsc.tws.dl.utils.Shape;
import edu.iu.dsc.tws.dl.utils.SingleShape;
import edu.iu.dsc.tws.dl.utils.Util;
import edu.iu.dsc.tws.dl.utils.pair.TensorArrayPair;

/**
 * The `forward(input)` reshape the input tensor into a
 * `size(0) * size(1) * ...` tensor, taking the elements row-wise.
 */
@SuppressWarnings("NeedBraces")
public class Reshape extends TensorModule<DenseTensor> {

  // size the reshape size
  private int[] size;

  private Boolean batchMode;
  //It is a optional argument. If it is set to `Some(true)`,
  //the first dimension of input is considered as batch dimension,
  //and thus keep this dimension size fixed. This is necessary
  //when dealing with batch sizes of one. When set to `Some(false)`,
  //it forces the entire input (including the first dimension) to be reshaped
  //to the input size. Default is `None`, which means the module considers
  //inputs with more elements than the product of provided sizes (size(0) *
  //size(1) * ..) to be batches, otherwise in no batch mode.
  private int[] batchSize;

  private int nElement = 1;

  // whether share the storage between input and output
  // in this layer, if input is contiguous, inplace is true. otherwise, inplace is false
  private boolean inPlace = true;

  public Reshape(int[] size) {
    this(size, null);
  }

  public Reshape(int[] size, Boolean mode) {
    this.output = new DenseTensor(this.isFloat);
    this.gradInput = new DenseTensor(this.isFloat);
    this.size = size;
    this.batchMode = mode;
    this.batchSize = new int[size.length + 1];
    for (int i = 1; i <= size.length; i++) {
      this.batchSize[i] = size[i - 1];
      nElement *= size[i - 1];
    }

  }

  @Override
  public DenseTensor updateOutput(DenseTensor input) {
    if ((batchMode != null && !batchMode)
        || (input.nElement() == nElement && batchMode == null && input.size(1) != 1)) {
      Util.require(input.nElement() == nElement,
          "element number must match Reshape size. "
              + "But In ${this.getName()} : element number is: ${ input.nElement() } , "
              + "reshape size is: ${nElement}");
      if (input.isContiguous()) {
        output = (DenseTensor) input.view(size);
      } else {
        output = (DenseTensor) input.contiguous().view(size);
        this.inPlace = false;
      }
    } else {
      Util.require(input.nElement() == nElement * input.size(1),
          "element number must match Reshape size. "
              + "But In ${this.getName()} : element number is: ${ input.nElement() } , "
              + "reshape size is: ${ nElement * input.size(1) }");
      batchSize[0] = input.size(1);
      if (input.isContiguous()) {
        output = (DenseTensor) input.view(batchSize);
      } else {
        output = (DenseTensor) input.contiguous().view(batchSize);
        this.inPlace = false;
      }
    }
    return (DenseTensor) output;
  }

  @Override
  public DenseTensor updateGradInput(DenseTensor input, DenseTensor gradOutput) {
    if (gradOutput.isContiguous()) {
      gradInput = (DenseTensor) gradOutput.view(input.size());
    } else {
      gradInput = (DenseTensor) gradOutput.contiguous().view(input.size());
    }
    return (DenseTensor) gradInput;
  }

  @Override
  public TensorArrayPair parameters() {
    return null;
  }

  @Override
  public Tensor[] getExtraParameter() {
    return null;
  }

  @Override
  public void reset() {

  }

  @Override
  public boolean equals(Object o) {
    if (this == o) return true;
    if (o == null || getClass() != o.getClass()) return false;
    if (!super.equals(o)) return false;
    Reshape reshape = (Reshape) o;
    return nElement == reshape.nElement && Arrays.equals(batchSize, reshape.batchSize);
  }

  @Override
  public int hashCode() {
    int seed = 37;
    int hash = super.hashCode();
    int i = 0;
    while (i < batchSize.length) {
      hash = hash * seed + batchSize[i];
      i += 1;
    }
    hash = hash * seed + nElement;
    hash = hash * seed + batchMode.hashCode();

    return hash;
  }

  @Override
  public AbstractModule clearState() {
    if (!this.inPlace) {
      super.clearState();
    }
    return this;
  }

  @Override
  public String toString() {
    return getPrintName() + "size=" + Arrays.toString(size) + '}';
  }

  @Override
  public Shape computeOutputShape(Shape inputShape) {
    int[] input = inputShape.toSingle();
    int[] output;
    if ((batchMode != null && !batchMode)
        || (TensorNumeric.product(input) == nElement && batchMode == null && input[0] != 1)) {
      output = size;
    } else {
      output = batchSize.clone();
      output[0] = input[0];
    }
    return new SingleShape(output);
  }
}
