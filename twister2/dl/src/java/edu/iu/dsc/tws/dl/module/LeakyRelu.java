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

import edu.iu.dsc.tws.dl.data.tensor.DenseTensor;
import edu.iu.dsc.tws.dl.graph.TensorModule;
import edu.iu.dsc.tws.dl.utils.Util;
import edu.iu.dsc.tws.dl.utils.pair.TensorArrayPair;

public class LeakyRelu extends TensorModule<DenseTensor> {
  private double negValue = 0.01;
  private boolean inplace = false;

  public LeakyRelu(double negValue) {
    this.negValue = negValue;
    this.output = new DenseTensor(this.isFloat);
    this.gradInput = new DenseTensor(this.isFloat);
    if (negValue < 0) {
      inplace = false;
    }
  }

  public LeakyRelu(double negValue, boolean inplace) {
    this.negValue = negValue;
    this.inplace = inplace;
    this.output = new DenseTensor(this.isFloat);
    this.gradInput = new DenseTensor(this.isFloat);
  }

  @Override
  public DenseTensor updateOutput(DenseTensor input) {
    Util.require(input.isContiguous(), "input should be contiguous");
    if (inplace) {
      output = input;
    }
    if (this.isFloat) {
      float negValuef = (float) negValue;
      if (inplace) {
        int i = input.storageOffset() - 1;
        float[] array = input.storage().toFloatArray();
        int end = input.nElement() + input.storageOffset() - 1;
        while (i < end) {
          if (array[i] < 0) {
            array[i] *= negValuef;
          }
          i += 1;
        }
      } else {
        ((DenseTensor) output).resizeAs(input);
        int i = 0;
        int inputOffset = input.storageOffset() - 1;
        float[] inputArray = input.storage().toFloatArray();
        int outputOffset = ((DenseTensor) output).storageOffset() - 1;
        float[] outputArray = ((DenseTensor) output).storage().toFloatArray();
        int end = input.nElement();
        while (i < end) {
          if (inputArray[i + inputOffset] < 0) {
            outputArray[i + outputOffset] = inputArray[i + inputOffset] * negValuef;
          } else {
            outputArray[i + outputOffset] = inputArray[i + inputOffset];
          }
          i += 1;
        }
      }
    } else {
      if (inplace) {
        int i = input.storageOffset() - 1;
        double[] array = input.storage().toDoubleArray();
        int end = input.nElement() + input.storageOffset() - 1;
        while (i < end) {
          if (array[i] < 0) {
            array[i] *= negValue;
          }
          i += 1;
        }
      } else {
        ((DenseTensor) output).resizeAs(input);
        int i = 0;
        int inputOffset = input.storageOffset() - 1;
        double[] inputArray = input.storage().toDoubleArray();
        int outputOffset = ((DenseTensor) output).storageOffset() - 1;
        double[] outputArray = ((DenseTensor) output).storage().toDoubleArray();
        int end = input.nElement();
        while (i < end) {
          if (inputArray[i + inputOffset] < 0) {
            outputArray[i + outputOffset] = inputArray[i + inputOffset] * negValue;
          } else {
            outputArray[i + outputOffset] = inputArray[i + inputOffset];
          }
          i += 1;
        }
      }
    }

    return (DenseTensor) output;
  }

  @Override
  public DenseTensor updateGradInput(DenseTensor input, DenseTensor gradOutput) {
    Util.require(input.isSameSizeAs(gradOutput),
        "input should have the same size with gradOutput"
            + "input size ${input.dim()} gradOutput size ${gradOutput.dim()}");
    Util.require(gradOutput.isContiguous(), "gradOutput should be contiguous");
    if (inplace) {
      gradInput = gradOutput;
    }
    if (this.isFloat) {
      float negValuef = (float) negValue;

      if (inplace) {
        int i = 0;
        int inputOffset = input.storageOffset() - 1;
        float[] inputArray = input.storage().toFloatArray();
        int gradInputOffset = ((DenseTensor) gradInput).storageOffset() - 1;
        float[] gradInputArray = ((DenseTensor) gradInput).storage().toFloatArray();
        int end = input.nElement();
        while (i < end) {
          if (inputArray[i + inputOffset] > 0) {
            gradInputArray[i + gradInputOffset] *= negValuef;
          }
          i += 1;
        }
      } else {
        ((DenseTensor) gradInput).resizeAs(input);
        int i = 0;
        int inputOffset = input.storageOffset() - 1;
        float[] inputArray = input.storage().toFloatArray();
        int gradOutputOffset = gradOutput.storageOffset() - 1;
        float[] gradOutputArray = gradOutput.storage().toFloatArray();
        int gradInputOffset = ((DenseTensor) gradInput).storageOffset() - 1;
        float[] gradInputArray = ((DenseTensor) gradInput).storage().toFloatArray();
        int end = input.nElement();
        while (i < end) {
          if (inputArray[i + inputOffset] < 0) {
            gradInputArray[i + gradInputOffset] = gradOutputArray[i + gradOutputOffset] * negValuef;
          } else {
            gradInputArray[i + gradInputOffset] = gradOutputArray[i + gradOutputOffset];
          }
          i += 1;
        }
      }
    } else {
      if (inplace) {
        int i = 0;
        int inputOffset = input.storageOffset() - 1;
        double[] inputArray = input.storage().toDoubleArray();
        int gradInputOffset = ((DenseTensor) gradInput).storageOffset() - 1;
        double[] gradInputArray = ((DenseTensor) gradInput).storage().toDoubleArray();
        int end = input.nElement();
        while (i < end) {
          if (inputArray[i + inputOffset] > 0) {
            gradInputArray[i + gradInputOffset] *= negValue;
          }
          i += 1;
        }
      } else {
        ((DenseTensor) gradInput).resizeAs(input);
        int i = 0;
        int inputOffset = input.storageOffset() - 1;
        double[] inputArray = input.storage().toDoubleArray();
        int gradOutputOffset = gradOutput.storageOffset() - 1;
        double[] gradOutputArray = gradOutput.storage().toDoubleArray();
        int gradInputOffset = ((DenseTensor) gradInput).storageOffset() - 1;
        double[] gradInputArray = ((DenseTensor) gradInput).storage().toDoubleArray();
        int end = input.nElement();
        while (i < end) {
          if (inputArray[i + inputOffset] < 0) {
            gradInputArray[i + gradInputOffset] = gradOutputArray[i + gradOutputOffset] * negValue;
          } else {
            gradInputArray[i + gradInputOffset] = gradOutputArray[i + gradOutputOffset];
          }
          i += 1;
        }
      }
    }

    return (DenseTensor) gradInput;
  }

  @Override
  public TensorArrayPair parameters() {
    return null;
  }

  @Override
  public void reset() {

  }

  @Override
  public AbstractModule clearState() {
    if (inplace) {
      super.clearState();
    }
    return this;
  }
}
