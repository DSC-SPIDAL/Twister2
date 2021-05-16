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
import edu.iu.dsc.tws.dl.utils.Shape;
import edu.iu.dsc.tws.dl.utils.Util;
import edu.iu.dsc.tws.dl.utils.pair.TensorArrayPair;

@SuppressWarnings("NeedBraces")
public class Threshold extends TensorModule<DenseTensor> {

  private double threshold = 1e-6;
  private double value = 0.0;
  private boolean inPlace = false;

  public Threshold() {
    this.output = new DenseTensor(this.isFloat);
    this.gradInput = new DenseTensor(this.isFloat);
    validateParameters();
  }

  public Threshold(double threshold, double value, boolean inPlace) {
    this.output = new DenseTensor(this.isFloat);
    this.gradInput = new DenseTensor(this.isFloat);
    this.threshold = threshold;
    this.value = value;
    this.inPlace = inPlace;
    validateParameters();
  }

  @Override
  public void toFloat() {
    super.toFloat();
    this.output = new DenseTensor(this.isFloat);
    this.gradInput = new DenseTensor(this.isFloat);
  }

  @Override
  public DenseTensor updateOutput(DenseTensor input) {
    Util.require(input.isContiguous());
    validateParameters();

    if (this.isFloat) {
      if (inPlace) {
        output = input;
        float[] inputData = input.storage().toFloatArray();
        int inputOffset = input.storageOffset() - 1;
        int taskSize = input.nElement();

        //TODO parallelize
        for (int i = 0; i < taskSize; i++) {
          if (inputData[inputOffset + i] <= this.threshold) {
            inputData[inputOffset + i] = (float) this.value;
          }
        }
        return input;
      } else {
        ((DenseTensor) output).resizeAs(input);

        float[] inputData = input.storage().toFloatArray();
        float[] outputData = ((DenseTensor) output).storage().toFloatArray();
        int inputOffset = input.storageOffset() - 1;
        int outputOffset = ((DenseTensor) output).storageOffset() - 1;
        int taskSize = input.nElement();
        //TODO parallelize
        for (int i = 0; i < taskSize; i++) {
          if (inputData[inputOffset + i] <= this.threshold) {
            outputData[outputOffset + i] = (float) this.value;
          } else {
            outputData[outputOffset + i] = inputData[inputOffset + i];
          }
        }
        return (DenseTensor) output;
      }
    } else {
      if (inPlace) {
        output = input;
        double[] inputData = input.storage().toDoubleArray();
        int inputOffset = input.storageOffset() - 1;
        int taskSize = input.nElement();

        //TODO parallelize
        for (int i = 0; i < taskSize; i++) {
          if (inputData[inputOffset + i] <= this.threshold) {
            inputData[inputOffset + i] = this.value;
          }
        }
        return input;
      } else {
        ((DenseTensor) output).resizeAs(input);

        double[] inputData = input.storage().toDoubleArray();
        double[] outputData = ((DenseTensor) output).storage().toDoubleArray();
        int inputOffset = input.storageOffset() - 1;
        int outputOffset = ((DenseTensor) output).storageOffset() - 1;
        int taskSize = input.nElement();
        //TODO parallelize
        for (int i = 0; i < taskSize; i++) {
          if (inputData[inputOffset + i] <= this.threshold) {
            outputData[outputOffset + i] = this.value;
          } else {
            outputData[outputOffset + i] = inputData[inputOffset + i];
          }
        }
        return (DenseTensor) output;
      }
    }
  }

  private DenseTensor updateGradInputNoContinuous(DenseTensor input, DenseTensor gradOutput) {
    validateParameters();
    throw new UnsupportedOperationException("updateGradInputNoContinuous not supported");
//    if (inPlace) {;
//      gradInput = gradOutput;
//      ev.getType() match {
//        case DoubleType =>
//          gradInput.map(input, (g, i) =>
//          if (i <= threshold) 0 else g)
//        case FloatType =>
//          gradInput.map(input, (g, i) =>
//          if (i <= threshold) 0 else g)
//        case _ =>
//          throw new UnsupportedOperationException("Only Float/Double supported");
//      }
//    }
//    else {
//      gradInput.resizeAs(gradOutput);
//      gradInput.copy(gradOutput);
//      ev.getType() match {
//        case DoubleType =>
//          gradInput.map(input, (g, i) =>
//          if (i > threshold) g else 0)
//        case FloatType =>
//          gradInput.map(input, (g, i) =>
//          if (i > threshold) g else 0)
//        case _ => throw new UnsupportedOperationException("Only Float/Double supported");
//      }
//    }
//    return gradInput;
  }

  @Override
  public DenseTensor updateGradInput(DenseTensor input, DenseTensor gradOutput) {
    validateParameters();

    int i = 1;
    while (i <= input.nDimension()) {
      if (input.stride(i) != gradOutput.stride(i)) {
        return updateGradInputNoContinuous(input, gradOutput);
      }
      i += 1;
    }

    if (this.isFloat) {
      if (inPlace) {
        gradInput = gradOutput;
        float[] gradInputData = ((DenseTensor) gradInput).storage().toFloatArray();
        int gradInputOffset = ((DenseTensor) gradInput).storageOffset() - 1;
        int taskSize = gradOutput.nElement();
        float[] inputData = input.storage().toFloatArray();
        int inputOffset = input.storageOffset() - 1;
        //TODO parallelize
        for (int j = 0; j < taskSize; j++) {
          if (inputData[inputOffset + j] <= this.threshold) {
            gradInputData[gradInputOffset + j] = 0.0f;
          }
        }
      } else {
        ((DenseTensor) gradInput).resizeAs(gradOutput);
        ((DenseTensor) gradInput).copy(gradOutput);
        float[] gradInputData = ((DenseTensor) gradInput).storage().toFloatArray();
        int gradInputOffset = ((DenseTensor) gradInput).storageOffset() - 1;
        int taskSize = gradOutput.nElement();

        float[] inputData = input.storage().toFloatArray();
        int inputOffset = input.storageOffset() - 1;
        //TODO parallelize
        for (int j = 0; j < taskSize; j++) {
          if (inputData[inputOffset + j] <= this.threshold) {
            gradInputData[gradInputOffset + j] = 0.0f;
          } else {
            gradInputData[gradInputOffset + j] = gradInputData[inputOffset + j];
          }
        }
      }
    } else {
      if (inPlace) {
        gradInput = gradOutput;
        double[] gradInputData = ((DenseTensor) gradInput).storage().toDoubleArray();
        int gradInputOffset = ((DenseTensor) gradInput).storageOffset() - 1;
        int taskSize = gradOutput.nElement();
        double[] inputData = input.storage().toDoubleArray();
        int inputOffset = input.storageOffset() - 1;
        //TODO parallelize
        for (int j = 0; j < taskSize; j++) {
          if (inputData[inputOffset + j] <= this.threshold) {
            gradInputData[gradInputOffset + j] = 0.0;
          }
        }
      } else {
        ((DenseTensor) gradInput).resizeAs(gradOutput);
        ((DenseTensor) gradInput).copy(gradOutput);
        double[] gradInputData = ((DenseTensor) gradInput).storage().toDoubleArray();
        int gradInputOffset = ((DenseTensor) gradInput).storageOffset() - 1;
        int taskSize = gradOutput.nElement();

        double[] inputData = input.storage().toDoubleArray();
        int inputOffset = input.storageOffset() - 1;
        //TODO parallelize
        for (int j = 0; j < taskSize; j++) {
          if (inputData[inputOffset + j] <= this.threshold) {
            gradInputData[gradInputOffset + j] = 0.0;
          } else {
            gradInputData[gradInputOffset + j] = gradInputData[inputOffset + j];
          }
        }
      }
    }
    return (DenseTensor) gradInput;
  }

  @Override
  public Shape computeOutputShape(Shape inputShape) {
    return inputShape;
  }

  @Override
  public TensorArrayPair parameters() {
    return null;
  }

  @Override
  public void reset() {

  }

  public void validateParameters() {
    if (inPlace) {
      Util.require(value <= threshold, "in-place processing requires value ("
          + value + "') not exceed threshold (" + threshold + ")");
    }
  }

  @Override
  public AbstractModule clearState() {
    if (!inPlace) {
      super.clearState();
    }
    return this;
  }

  @Override
  public String toString() {
    return this.getPrintName() + "(" + this.threshold + ", " + this.value + ")";
  }

  @Override
  public boolean equals(Object o) {
    if (this == o) return true;
    if (o == null || getClass() != o.getClass()) return false;
    if (!super.equals(o)) return false;
    Threshold threshold1 = (Threshold) o;
    return Double.compare(threshold1.threshold, threshold) == 0
        && Double.compare(threshold1.value, value) == 0
        && inPlace == threshold1.inPlace;
  }

  @Override
  public int hashCode() {
    int seed = 37;
    int hash = super.hashCode();
    hash = hash * seed + (int) threshold;
    hash = hash * seed + (int) value;
    hash = hash * seed + ((Boolean) inPlace).hashCode();

    return hash;
  }
}
