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

import edu.iu.dsc.tws.dl.data.Tensor;
import edu.iu.dsc.tws.dl.data.tensor.DenseTensor;
import edu.iu.dsc.tws.dl.graph.TensorModule;
import edu.iu.dsc.tws.dl.optim.Initializable;
import edu.iu.dsc.tws.dl.optim.InitializationMethod;
import edu.iu.dsc.tws.dl.optim.Regularizer;
import edu.iu.dsc.tws.dl.optim.initialization.Zeros;
import edu.iu.dsc.tws.dl.utils.ErrorConstants;
import edu.iu.dsc.tws.dl.utils.Util;
import edu.iu.dsc.tws.dl.utils.pair.TensorArrayPair;
import edu.iu.dsc.tws.dl.utils.varformat.OUT_IN;

/**
 * The `Linear` module applies a linear transformation to the input data,
 * i.e. `y = Wx + b`. The `input` given in `forward(input)` must be either
 * a vector (1D tensor) or matrix (2D tensor). If the input is a vector, it must
 * have the size of `inputSize`. If it is a matrix, then each row is assumed to be
 * an input sample of given batch (the number of rows means the batch size and
 * the number of columns should be equal to the `inputSize`).
 */
@SuppressWarnings("NeedBraces")
public class Linear extends TensorModule<DenseTensor> implements Initializable {

  protected InitializationMethod weightInitMethod = new Zeros();
  protected InitializationMethod biasInitMethod = new Zeros();
  //the size the each input sample
  private int inputSize;
  //the size of the module ((DenseTensor output) of each sample
  private int outputSize;
  //instance of [[Regularizer]]
  // (eg. L1 or L2 regularization), applied to the input weights matrices.
  private Regularizer wRegularizer;
  //instance of [[Regularizer]]
  //applied to the bias.
  private Regularizer bRegularizer;
  private boolean withBias = true;
  private Tensor initWeight;
  private Tensor initBias;
  private Tensor initGradWeight;
  private Tensor initGradBias;
  private Tensor weight;
  private Tensor bias;
  private Tensor addBuffer;
  private Tensor gradWeight;
  private Tensor gradBias;


  public Linear(int inputSize, int outputSize) {
    this(inputSize, outputSize, null, null, false);
  }

  public Linear(int inputSize, int outputSize, boolean isF) {
    this(inputSize, outputSize, null, null, isF);
  }

  public Linear(int inputSize, int outputSize, Regularizer wRegularizer, Regularizer bRegularizer,
                boolean isF) {
    this(inputSize, outputSize, wRegularizer, bRegularizer, true, null,
        null, null, null, isF);
  }

  public Linear(int inputSize, int outputSize, Regularizer wRegularizer, Regularizer bRegularizer,
                boolean withBias, Tensor initWeight, Tensor initBias, Tensor initGradWeight,
                Tensor initGradBias, boolean isF) {
    this.inputSize = inputSize;
    this.outputSize = outputSize;
    this.wRegularizer = wRegularizer;
    this.bRegularizer = bRegularizer;
    this.withBias = withBias;
    this.initWeight = initWeight;
    this.initBias = initBias;
    this.initGradWeight = initGradWeight;
    this.initGradBias = initGradBias;
    this.isFloat = isF;
    init();

  }

  private void init() {
    this.output = new DenseTensor(this.isFloat);
    this.gradInput = new DenseTensor(this.isFloat);
    weight = (initWeight != null) ? initWeight : new DenseTensor(outputSize,
        inputSize, this.isFloat);

    if (initBias != null) {
      bias = initBias;
    } else if (withBias) {
      bias = new DenseTensor(outputSize, this.isFloat);
    } else {
      bias = null;
    }
    addBuffer = new DenseTensor(this.isFloat);

    gradWeight = (initGradWeight != null) ? initGradWeight : new DenseTensor(this.isFloat);

    if (initGradBias != null) {
      gradBias = initGradBias;
    } else if (withBias) {
      gradBias = new DenseTensor(this.isFloat);
    } else {
      bias = null;
    }

    double stdv = 1.0 / Math.sqrt(weight.size(2));
    //weightInitMethod = new RandomUniform(-stdv, stdv);
    //biasInitMethod = new RandomUniform(-stdv, stdv);
    reset();
  }

  @Override
  public DenseTensor updateOutput(DenseTensor input) {
    Util.require(input.dim() == 1 || input.dim() == 2,
        "Linear: " + ErrorConstants.constrainInputAsVectorOrBatch
            + "input dim ${input.dim()}");

    if (this.isFloat) {
      if (input.dim() == 1) {
        ((DenseTensor) output).resize(new int[]{outputSize});
        if (withBias) {
          ((DenseTensor) output).copy(bias);
        } else {
          ((DenseTensor) output).zero();
        }

        ((DenseTensor) output).addmv(1f, weight, input);
      } else if (input.dim() == 2) {
        int nFrame = input.size(1);
        int nElement = ((DenseTensor) output).nElement();
        int[] t = new int[]{nFrame, weight.size(1)};
        ((DenseTensor) output).resize(t);
        if (((DenseTensor) output).nElement() != nElement) {
          ((DenseTensor) output).zero();
        }

        if (addBuffer.nElement() != nFrame) {
          addBuffer.resize(new int[]{nFrame}).fill(1.0f);
        }

        ((DenseTensor) output).addmm(0.0f, (DenseTensor) output, 1.0f, input, weight.t());
        if (withBias) ((DenseTensor) output).addr(1.0f, addBuffer, bias);
      }
    } else {
      if (input.dim() == 1) {
        ((DenseTensor) output).resize(new int[]{outputSize});
        if (withBias) {
          ((DenseTensor) output).copy(bias);
        } else {
          ((DenseTensor) output).zero();
        }

        ((DenseTensor) output).addmv(1, weight, input);
      } else if (input.dim() == 2) {
        int nFrame = input.size(1);
        int nElement = ((DenseTensor) output).nElement();
        int[] t = new int[]{nFrame, weight.size(1)};
        ((DenseTensor) output).resize(t);
        if (((DenseTensor) output).nElement() != nElement) {
          ((DenseTensor) output).zero();
        }

        if (addBuffer.nElement() != nFrame) {
          addBuffer.resize(new int[]{nFrame}).fill(1.0);
        }

        ((DenseTensor) output).addmm(0.0, (DenseTensor) output, 1.0, input, weight.t());
        if (withBias) ((DenseTensor) output).addr(1.0, addBuffer, bias);
      }
    }

    return (DenseTensor) output;
  }

  @Override
  public DenseTensor updateGradInput(DenseTensor input, DenseTensor gradOutput) {
    Util.require(input.dim() == 1 || input.dim() == 2,
        "Linear: " + ErrorConstants.constrainInputAsVectorOrBatch
            + "input dim ${input.dim()}");

    int nElement = ((DenseTensor) gradInput).nElement();
    ((DenseTensor) gradInput).resizeAs(input);
    if (nElement != ((DenseTensor) gradInput).nElement()) {
      ((DenseTensor) gradInput).zero();
    }

    if (this.isFloat) {
      if (input.dim() == 1) {
        ((DenseTensor) gradInput).addmv(0.0f, 1.0f, weight.t(), gradOutput);
      } else if (input.dim() == 2) {
        ((DenseTensor) gradInput).addmm(0.0f, 1.0f, gradOutput, weight);
      }
    } else {
      if (input.dim() == 1) {
        ((DenseTensor) gradInput).addmv(0.0, 1.0, weight.t(), gradOutput);
      } else if (input.dim() == 2) {
        ((DenseTensor) gradInput).addmm(0.0, 1.0, gradOutput, weight);
      }
    }
    return (DenseTensor) gradInput;
  }

  @Override
  public TensorArrayPair parameters() {
    if (null == bias) {
      return new TensorArrayPair(new Tensor[]{this.weight}, new Tensor[]{this.gradWeight});
    } else {
      return new TensorArrayPair(new Tensor[]{this.weight, this.bias},
          new Tensor[]{this.gradWeight, this.gradBias});
    }
  }

  @Override
  public void accGradParameters(DenseTensor input, DenseTensor gradOutput) {
    Util.require(input.dim() == 1 || input.dim() == 2,
        "Linear: " + ErrorConstants.constrainInputAsVectorOrBatch
            + "input dim ${input.dim()}");

    gradWeight.resize(outputSize, inputSize);
    if (withBias) {
      gradBias.resize(outputSize);
    }
    if (this.isFloat) {
      float scaleWf = (float) scaleW;
      float scaleBf = (float) scaleB;
      if (input.dim() == 1) {
        if (scaleW != 0) {
          gradWeight.addr(scaleWf, gradOutput, input);
        }

        if (withBias && scaleBf != 0) {
          gradBias.add(scaleBf, gradOutput);
        }
      } else if (input.dim() == 2) {
        if (scaleW != 0) {
          gradWeight.addmm(scaleWf, gradOutput.t(), input);
        }

        if (withBias && scaleBf != 0) {
          gradBias.addmv(scaleBf, gradOutput.t(), addBuffer);
        }
      }
      if (null != wRegularizer && scaleW != 0) {
        wRegularizer.accRegularization(weight, gradWeight, scaleWf);
      }
      if (null != bRegularizer && scaleB != 0) {
        bRegularizer.accRegularization(bias, gradBias, scaleBf);
      }
    } else {
      if (input.dim() == 1) {
        if (scaleW != 0) {
          gradWeight.addr(scaleW, gradOutput, input);
        }

        if (withBias && scaleB != 0) {
          gradBias.add(scaleB, gradOutput);
        }
      } else if (input.dim() == 2) {
        if (scaleW != 0) {
          gradWeight.addmm(scaleW, gradOutput.t(), input);
        }

        if (withBias && scaleB != 0) {
          gradBias.addmv(scaleB, gradOutput.t(), addBuffer);
        }
      }
      if (null != wRegularizer && scaleW != 0) {
        wRegularizer.accRegularization(weight, gradWeight, scaleW);
      }
      if (null != bRegularizer && scaleB != 0) {
        bRegularizer.accRegularization(bias, gradBias, scaleB);
      }
    }
  }

  @Override
  public Tensor[] getExtraParameter() {
    return null;
  }

  @Override
  public void reset() {
    if (initWeight == null) {
      weightInitMethod.init(weight, new OUT_IN());
    }
    if (initBias == null && bias != null) {
      biasInitMethod.init(bias, new OUT_IN());
    }
    zeroGradParameters();
  }

  @Override
  public String toString() {
    return this.getPrintName() + "(" + inputSize + "->" + outputSize + ")";
  }

  @Override
  public AbstractModule clearState() {
    super.clearState();
    addBuffer.set();
    return this;
  }

  @Override
  public Initializable setInitMethod(InitializationMethod weightMethod,
                                     InitializationMethod biasMethod) {
    if (weightInitMethod != null) {
      this.weightInitMethod = weightMethod;
    }

    if (biasInitMethod != null) {
      this.biasInitMethod = biasMethod;
    }
    reset();
    return this;
  }
}
