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
import edu.iu.dsc.tws.dl.utils.RandomGenerator;
import edu.iu.dsc.tws.dl.utils.Shape;
import edu.iu.dsc.tws.dl.utils.pair.TensorArrayPair;

/**
 * Dropout masks(set to zero) parts of input using a bernoulli distribution.
 * Each input element has a probability initP of being dropped. If `scale` is
 * true(true by default), the outputs are scaled by a factor of `1/(1-initP)`
 * during training.
 * During eintuating, output is the same as input.
 * <p>
 * It has been proven an effective approach for regularization and preventing
 * co-adaptation of feature detectors. For more details, plese see
 * [Improving neural networks by preventing co-adaptation of feature detectors]
 * (https://arxiv.org/abs/1207.0580)
 *
 * @param initP   the probability p
 * @param inplace whether to make `input` and `output` share the same storage
 * @param scale   whether to scale the output by a factor of `1 / (1 - p)`
 */
public class Dropout extends TensorModule<DenseTensor> {
  private double initP = 0.5;
  private float initPf = 0.5f;
  private boolean inplace = false;
  private boolean scale = true;
  private DenseTensor noise = new DenseTensor(false);
  private boolean isResampling = true;

  public Dropout() {
  }

  public Dropout(double initP, boolean isF) {
    this.initP = initP;
    this.isFloat = isF;
    this.initPf = (float) initP;
    this.output = new DenseTensor(this.isFloat);
    this.gradInput = new DenseTensor(this.isFloat);
  }

  public Dropout(double initP, boolean inplace, boolean scale, boolean isF) {
    this.initP = initP;
    this.initPf = (float) initP;
    this.inplace = inplace;
    this.scale = scale;
    this.isFloat = isF;
    this.output = new DenseTensor(this.isFloat);
    this.gradInput = new DenseTensor(this.isFloat);
  }

  @Override
  public void toFloat() {
    super.toFloat();
    noise = new DenseTensor(true);
  }

  @Override
  public DenseTensor updateOutput(DenseTensor input) {
    if (inplace) {
      this.output = input;
    } else {
      ((DenseTensor) this.output).resizeAs(input).copy(input);
    }

    if (train) {
      noise.resizeAs(input);
      if (input.isContiguous()) {
        if (isResampling) {
          if (this.isFloat) {
            float[] noiseData = noise.storage().toFloatArray();
            int taskSize = noise.nElement();
            int extraTask = noise.nElement();
            int allocated = 0;
            int offset = ((DenseTensor) this.output).storageOffset() - 1;
            float[] data = ((DenseTensor) this.output).storage().toFloatArray();
            //TODO parallelize
            for (int k = 0; k < taskSize; k++) {
              if (RandomGenerator.RNG().bernoulli(1 - initPf)) {
                if (scale) {
                  data[offset + k] = TensorNumeric.divide(data[offset + k], 1.0f - initPf);
                  noiseData[k] = (float) (1.0 / (1 - initPf));
                } else {
                  noiseData[k] = 1;
                }
              } else {
                data[offset + k] = 0;
                noiseData[k] = 0;
              }
            }
          } else {
            double[] noiseData = noise.storage().toDoubleArray();
            int taskSize = noise.nElement();
            int extraTask = noise.nElement();
            int allocated = 0;
            int offset = ((DenseTensor) this.output).storageOffset() - 1;
            double[] data = ((DenseTensor) this.output).storage().toDoubleArray();
            //TODO parallelize
            for (int k = 0; k < taskSize; k++) {
              if (RandomGenerator.RNG().bernoulli(1 - initP)) {
                if (scale) {
                  data[offset + k] = TensorNumeric.divide(data[offset + k], 1.0 - initP);
                  noiseData[k] = 1.0 / (1 - initP);
                } else {
                  noiseData[k] = 1;
                }
              } else {
                data[offset + k] = 0;
                noiseData[k] = 0;
              }
            }
          }
        } else {
          ((DenseTensor) this.output).cmul(noise);
        }
        return (DenseTensor) this.output;
      } else {
        if (this.isFloat) {
          if (isResampling) {
            noise.bernoulli(1f - initPf);

            if (scale) {
              noise.div(1.0f - initPf);
            }
          }
        } else {
          if (isResampling) {
            noise.bernoulli(1 - initP);

            if (scale) {
              noise.div(1.0 - initP);
            }
          }
        }


        return (DenseTensor) ((DenseTensor) this.output).cmul(noise);
      }
    } else if (!scale) {
      if (this.isFloat) {
        return (DenseTensor) ((DenseTensor) this.output).mul(1.0f - initPf);
      } else {
        return (DenseTensor) ((DenseTensor) this.output).mul(1.0 - initP);
      }
    } else {
      return (DenseTensor) output;
    }
  }

  @Override
  public DenseTensor updateGradInput(DenseTensor input, DenseTensor gradOutput) {
    if (train) {
      if (inplace) {
        this.gradInput = gradOutput;
      } else {
        ((DenseTensor) this.gradInput).resizeAs(gradOutput).copy(gradOutput);
      }

      if (((DenseTensor) gradInput).isContiguous()) {
        if (this.isFloat) {
          float[] noiseData = noise.storage().toFloatArray();
          int taskSize = noise.nElement();
          int extraTask = noise.nElement();
          float[] gradInputData = ((DenseTensor) gradInput).storage().toFloatArray();
          int gradInputOffset = ((DenseTensor) gradInput).storageOffset() - 1;
          int allocated = 0;
          for (int k = 0; k < taskSize; k++) {
            gradInputData[gradInputOffset + k] =
                TensorNumeric.times(gradInputData[gradInputOffset + k], noiseData[k]);
          }
        } else {
          double[] noiseData = noise.storage().toDoubleArray();
          int taskSize = noise.nElement();
          int extraTask = noise.nElement();
          double[] gradInputData = ((DenseTensor) gradInput).storage().toDoubleArray();
          int gradInputOffset = ((DenseTensor) gradInput).storageOffset() - 1;
          int allocated = 0;
          for (int k = 0; k < taskSize; k++) {
            gradInputData[gradInputOffset + k] =
                TensorNumeric.times(gradInputData[gradInputOffset + k], noiseData[k]);
          }
        }
      } else {
        ((DenseTensor) this.gradInput).cmul(noise);
      }
    } else {
      throw new IllegalArgumentException("backprop only defined while training");
    }

    return (DenseTensor) this.gradInput;
  }

  @Override
  public AbstractModule clearState() {
    if (!inplace) {
      super.clearState();
    }
    noise.set();
    return this;
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
}
