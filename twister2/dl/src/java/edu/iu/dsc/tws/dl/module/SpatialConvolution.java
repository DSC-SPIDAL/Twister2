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
import edu.iu.dsc.tws.dl.data.format.DataFormat;
import edu.iu.dsc.tws.dl.data.format.NCHW;
import edu.iu.dsc.tws.dl.data.format.NHWC;
import edu.iu.dsc.tws.dl.data.tensor.DenseTensor;
import edu.iu.dsc.tws.dl.graph.TensorModule;
import edu.iu.dsc.tws.dl.optim.Initializable;
import edu.iu.dsc.tws.dl.optim.InitializationMethod;
import edu.iu.dsc.tws.dl.optim.Regularizer;
import edu.iu.dsc.tws.dl.optim.initialization.RandomUniform;
import edu.iu.dsc.tws.dl.optim.initialization.Zeros;
import edu.iu.dsc.tws.dl.utils.ErrorConstants;
import edu.iu.dsc.tws.dl.utils.NNPrimitive;
import edu.iu.dsc.tws.dl.utils.Shape;
import edu.iu.dsc.tws.dl.utils.SingleShape;
import edu.iu.dsc.tws.dl.utils.Util;
import edu.iu.dsc.tws.dl.utils.VariableFormat;
import edu.iu.dsc.tws.dl.utils.pair.TensorArrayPair;
import edu.iu.dsc.tws.dl.utils.varformat.GP_KH_KW_IN_OUT;
import edu.iu.dsc.tws.dl.utils.varformat.GP_OUT_IN_KW_KH;
import edu.iu.dsc.tws.dl.utils.varformat.ONE_D;

@SuppressWarnings({"HiddenField", "LocalVariableName", "NeedBraces", "MemberName"})
public class SpatialConvolution extends TensorModule<DenseTensor> implements Initializable {

  protected InitializationMethod weightInitMethod = new Zeros();
  protected InitializationMethod biasInitMethod = new Zeros();

  // The number of expected input planes in the image given private into forward()
  private int nInputPlane;
  private int nOutputPlane; // The number of output planes the convolution layer will produce.
  private int kernelW; // The kernel width of the convolution
  private int kernelH; // The kernel height of the convolution
  private int strideW = 1; // The step of the convolution in the width dimension.
  private int strideH = 1; // The step of the convolution in the height dimension
  private int padW = 0; // The additional zeros added per width to the input planes.
  private int padH = 0; // The additional zeros added per height to the input planes.
  private int nGroup = 1; // Kernel group number
  private boolean propagateBack = true; // propagate gradient back
  private Regularizer wRegularizer = null;
  private Regularizer bRegularizer = null;
  private Tensor initWeight = null;
  private Tensor initBias = null;
  private Tensor initGradWeight = null;
  private Tensor initGradBias = null;
  private boolean withBias = true;
  private DataFormat format = new NCHW();

  private DenseTensor weight;
  private DenseTensor bias;
  private DenseTensor gradWeight;
  private DenseTensor gradBias;

  private int[] weightMMShape;
  private VariableFormat weightFormat;

  private DenseTensor fInput = new DenseTensor(false);
  private DenseTensor fGradInput = new DenseTensor(false);
  protected DenseTensor ones = new DenseTensor(false);
  protected DenseTensor onesBatch = new DenseTensor(false);
  protected DenseTensor onesBias = null;
  protected DenseTensor weightMM = null;
  protected DenseTensor gradientBiasMT = null;
  protected DenseTensor gradWeightMM = null;
  protected transient DenseTensor gradWeightMMInBatch = null;
  protected boolean _1x1;

  protected long im2colTime = 0L;
  protected long col2imTime = 0L;

  public SpatialConvolution(int nInputPlane, int nOutputPlane, int kernelW, int kernelH,
                            int strideW, int strideH, int padW, int padH, int nGroup,
                            boolean propagateBack, Regularizer wRegularizer,
                            Regularizer bRegularizer, boolean isF) {
    this(null, null, nInputPlane, nOutputPlane, kernelW, kernelH,
        strideW, strideH, padW, padH,
        nGroup, propagateBack, wRegularizer, bRegularizer, null, null, null,
        null, true, new NCHW(), isF);
  }

  public SpatialConvolution(int nInputPlane, int nOutputPlane, int kernelW, int kernelH,
                            int strideW, int strideH, int padW, int padH, int nGroup,
                            boolean propagateBack, Regularizer wRegularizer,
                            Regularizer bRegularizer, Tensor initWeight,
                            Tensor initBias, Tensor initGradWeight, Tensor initGradBias,
                            boolean withBias, DataFormat format, boolean isF) {
    this(null, null, nInputPlane, nOutputPlane, kernelW, kernelH,
        strideW, strideH, padW, padH,
        nGroup, propagateBack, wRegularizer, bRegularizer, initWeight, initBias, initGradWeight,
        initGradBias, withBias, format, isF);
  }

  @Override
  public void toFloat() {
    super.toFloat();
    fInput = new DenseTensor(true);
    fGradInput = new DenseTensor(true);
    ones = new DenseTensor(true);
    onesBatch = new DenseTensor(true);
    this.output = new DenseTensor(this.isFloat);
    this.gradInput = new DenseTensor(this.isFloat);
  }

  /**
   * SpatialConvolution.
   *
   * @param weightInitMethod
   * @param biasInitMethod
   * @param nInputPlane
   * @param nOutputPlane
   * @param kernelW
   * @param kernelH
   * @param strideW
   * @param strideH
   * @param padW
   * @param padH
   * @param nGroup
   * @param propagateBack
   * @param wRegularizer
   * @param bRegularizer
   * @param initWeight
   * @param initBias
   * @param initGradWeight
   * @param initGradBias
   * @param withBias
   * @param format
   */
  public SpatialConvolution(InitializationMethod weightInitMethod,
                            InitializationMethod biasInitMethod, int nInputPlane, int nOutputPlane,
                            int kernelW, int kernelH, int strideW, int strideH, int padW, int padH,
                            int nGroup, boolean propagateBack, Regularizer wRegularizer,
                            Regularizer bRegularizer, Tensor initWeight, Tensor initBias,
                            Tensor initGradWeight, Tensor initGradBias, boolean withBias,
                            DataFormat format, boolean isF) {

    //this.weightInitMethod = weightInitMethod;
    //this.biasInitMethod = biasInitMethod;
    this.nInputPlane = nInputPlane;
    this.nOutputPlane = nOutputPlane;
    this.kernelW = kernelW;
    this.kernelH = kernelH;
    this.strideW = strideW;
    this.strideH = strideH;
    this.padW = padW;
    this.padH = padH;
    this.nGroup = nGroup;
    this.propagateBack = propagateBack;
    this.wRegularizer = wRegularizer;
    this.bRegularizer = bRegularizer;
    this.initWeight = initWeight;
    this.initBias = initBias;
    this.initGradWeight = initGradWeight;
    this.initGradBias = initGradBias;
    this.withBias = withBias;
    this.format = format;
    this.isFloat = isF;

    Util.require(nOutputPlane % nGroup == 0, "Number of input channels "
        + "should be multiples of group "
        + "number of input channels ${nInputPlane}, "
        + "group ${nGroup}.");
    Util.require(nOutputPlane % nGroup == 0,
        "Number of output channels "
            + "should be multiples of group "
            + "(number of output channels ${nOutputPlane}, "
            + "group ${nGroup}).");

    if (nGroup != 1) {
      Util.require(format instanceof NCHW, "group convolution is not supported in NHWC format");
    }
    Util.require((padW >= 0 && padH >= 0) || (padW == -1 && padH == -1),
        "Illegal padding configuration (padW: $padW, padH: $padH)");
    this.output = new DenseTensor(this.isFloat);
    this.gradInput = new DenseTensor(this.isFloat);

    if (initWeight != null) {
      this.weight = (DenseTensor) initWeight;
    } else {
      if (format instanceof NCHW) {
        this.weight = new DenseTensor(new int[]{nGroup, nOutputPlane / nGroup,
            nInputPlane * kernelH * kernelW / nGroup}, this.isFloat);
      } else if (format instanceof NHWC) {
        this.weight = new DenseTensor(1, nInputPlane * kernelH * kernelW,
            nOutputPlane, this.isFloat);
      }
    }

    if (!withBias) {
      bias = null;
    } else if (initBias != null) {
      bias = (DenseTensor) initBias;
    } else {
      bias = new DenseTensor(nOutputPlane, this.isFloat);
    }

    if (initGradWeight != null) {
      gradWeight = (DenseTensor) initGradWeight;
    } else {
      if (format instanceof NCHW) {
        this.gradWeight = new DenseTensor(new int[]{nGroup, nOutputPlane / nGroup,
            nInputPlane * kernelH * kernelW / nGroup}, this.isFloat);
      } else if (format instanceof NHWC) {
        this.gradWeight = new DenseTensor(1, nInputPlane * kernelH * kernelW,
            nOutputPlane, this.isFloat);
      }
    }

    if (!withBias) {
      gradBias = null;
    } else if (initBias != null) {
      gradBias = (DenseTensor) initGradBias;
    } else {
      gradBias = new DenseTensor(nOutputPlane, this.isFloat);
    }

    if (format instanceof NCHW) {
      this.weightMMShape = new int[]{nGroup, nOutputPlane / nGroup,
          nInputPlane * kernelH * kernelW / nGroup};
      this.weightFormat = new GP_OUT_IN_KW_KH();
    } else if (format instanceof NHWC) {
      this.weightMMShape = new int[]{1, nInputPlane * kernelH * kernelW, nOutputPlane};
      this.weightFormat = new GP_KH_KW_IN_OUT();
    }

    if (withBias) {
      this.onesBias = new DenseTensor(this.isFloat);
      this.gradientBiasMT = new DenseTensor(this.isFloat);
    }

    if (kernelH == 1 && kernelW == 1 && strideW == 1 && strideH == 1
        && padH == 0 && padW == 0) {
      this._1x1 = true;
    } else {
      this._1x1 = false;
    }

    double stdv = 1.0 / Math.sqrt(kernelW * kernelH * nInputPlane);
    InitializationMethod wInit = new RandomUniform(-stdv, stdv);
    InitializationMethod bInit;
    if (withBias) {
      bInit = new RandomUniform(-stdv, stdv);
    } else {
      bInit = null;
    }
    if (weightInitMethod != null) {
      wInit = weightInitMethod;
    }
    if (biasInitMethod != null) {
      bInit = biasInitMethod;
    }
    //TODO remove
    wInit = new Zeros();
    bInit = new Zeros();
    setInitMethod(wInit, bInit);
  }

  public long getIm2colTime() {
    return im2colTime;
  }

  public long getCol2imTime() {
    return col2imTime;
  }


  @Override
  public void reset() {
    if (initWeight == null) {
      weightInitMethod.init(weight, weightFormat);
    }
    if (withBias && initBias == null) {
      biasInitMethod.init(bias, new ONE_D());
    }
    zeroGradParameters();
  }

  @Override
  public Shape computeOutputShape(Shape inputShape) {
    int[] input = inputShape.toSingle();
    Util.require(input.length == 4,
        "Convolution2D requires 4D input, but got input dim ${input.length}");
    //(dimHeight, dimWidth, channelDim)
    int[] dimHWD = format.getHWCDims(input.length);
    Util.require(input[dimHWD[2] - 1] == nInputPlane, "input channel size "
        + "${input(channelDim -1)} is not the same as nInputPlane $nInputPlane");
    int inputWidth = input[dimHWD[1] - 1];
    int inputHeight = input[dimHWD[0] - 1];
    int[] sizes = new int[0];
    if (padW == -1 && padH == -1) {
      sizes = Util.getSAMEOutSizeAndPadding(inputHeight, inputWidth, strideH,
          strideW, kernelH, kernelW);
    } else {
      Util.getOutSizeAndPadding(inputHeight, inputWidth, strideH, strideW,
          kernelH, kernelW, padH, padW, false);
    }
    int outputHeight = sizes[4];
    int outputWidth = sizes[5];
    Util.require(outputWidth >= 1 && outputHeight >= 1,
        "output size is too small. outputWidth: $outputWidth, outputHeight: $outputHeight");
    int[] outputShape = getOutputShape(outputHeight, outputWidth, input[0]);
    return new SingleShape(outputShape);
  }

  // batchSize = -2 by default means no batch. -1 represents batch in shape inference
  private int[] getOutputShape(int oh, int ow, int batchSize) {

    if (format instanceof NCHW) {
      if (batchSize == -2) {
        return new int[]{nOutputPlane, oh, ow};
      } else {
        return new int[]{batchSize, nOutputPlane, oh, ow};
      }
    } else if (format instanceof NHWC) {
      if (batchSize == -2) {
        return new int[]{oh, ow, nOutputPlane};
      } else {
        return new int[]{batchSize, oh, ow, nOutputPlane};
      }
    }
    return null;
  }

  // batchSize = -1 by default means no batch. -1 represents batch in shape inference
  private int[] getFInputShape(int oh, int ow, int batchSize) {

    if (format instanceof NCHW) {
      if (batchSize == -1) {
        return new int[]{nGroup, kernelW * kernelH * nInputPlane / nGroup, oh * ow};
      } else {
        return new int[]{batchSize, nGroup, kernelW * kernelH * nInputPlane / nGroup, oh * ow};
      }
    } else if (format instanceof NHWC) {
      if (batchSize == -1) {
        return new int[]{1, oh * ow, kernelW * kernelH * nInputPlane};
      } else {
        return new int[]{batchSize, 1, oh * ow, kernelW * kernelH * nInputPlane};
      }
    }
    return null;
  }

  // return (padTop, padDown, padLeft, padRight)
  protected int[] getPadding(int inputHeight, int inputWidth) {
    if (padW == -1 && padH == -1) {
      // deal with SAME padding
      int oW = (int) Math.ceil((double) inputWidth / (double) strideW);
      int oH = (int) Math.ceil((double) inputHeight / (double) strideH);
      int padAlongWidth = Math.max(0, (oW - 1) * strideW + kernelW - inputWidth);
      int padAlongHeight = Math.max(0, (oH - 1) * strideH + kernelH - inputHeight);
      return new int[]{padAlongHeight / 2, padAlongHeight - padAlongHeight / 2,
          padAlongWidth / 2, padAlongWidth - padAlongWidth / 2};
    } else {
      return new int[]{padH, padH, padW, padW};
    }
  }

  @Override
  public DenseTensor updateOutput(DenseTensor input) {
    Util.require(input.dim() == 3 || input.dim() == 4,
        "SpatialConvolution: " + ErrorConstants.constrainInputAs3DOrBatch);
    Util.require(input.isContiguous());
    long startTimef = System.nanoTime();

    if (weightMM == null || weightMM.storage().isEmpty()) {
      weightMM = (DenseTensor) weight.view(weightMMShape);
    }

    //(dimHeight, dimWidth, channelDim)
    int[] dimHWD = format.getHWCDims(input.dim());
    Util.require(input.size(dimHWD[2]) == nInputPlane, "input channel size "
        + "${input.size(channelDim)} is not the same as nInputPlane $nInputPlane");

    int inputWidth = input.size(dimHWD[1]);
    int inputHeight = input.size(dimHWD[0]);

    int[] sizes;
    if (padW == -1 && padH == -1) {
      sizes = Util.getSAMEOutSizeAndPadding(inputHeight, inputWidth, strideH,
          strideW, kernelH, kernelW);
    } else {
      sizes = Util.getOutSizeAndPadding(inputHeight, inputWidth, strideH, strideW,
          kernelH, kernelW, padH, padW, false);
    }

    int padTop = sizes[0];
    int padBottom = sizes[1];
    int padLeft = sizes[2];
    int padRight = sizes[3];
    int outputHeight = sizes[4];
    int outputWidth = sizes[5];

    Util.require(outputWidth >= 1 && outputHeight >= 1,
        "output size is too small. outputWidth: $outputWidth, outputHeight: $outputHeight");

    if (withBias && (onesBias.dim() != 1 || onesBias.size(1) != outputHeight * outputWidth)) {
      if (this.isFloat) {
        onesBias.resize(new int[]{outputHeight * outputWidth}).fill(1.0f);
      } else {
        onesBias.resize(new int[]{outputHeight * outputWidth}).fill(1.0);
      }
    }
  //  System.out.println("SpatialConv Time 1 : " + (System.nanoTime() - startTimef) / 1e6);

    if (input.dim() == 3) {
      Util.require(input.isContiguous());
      ((DenseTensor) output).resize(getOutputShape(outputHeight, outputWidth, -2));
      if (_1x1) {
        fInput.set(input);
        fInput.resize(getFInputShape(outputHeight, outputWidth, -1));
      } else {
        fInput.resize(getFInputShape(outputHeight, outputWidth, -1));
      }
     // System.out.println("SpatialConv3 Time 2 : " + (System.nanoTime() - startTimef) / 1e6);

      int g = 0;
      while (g < nGroup) {
        DenseTensor biasUse = null;
        if (withBias) {
          biasUse = (DenseTensor) bias.narrow(1, g * nOutputPlane / nGroup + 1,
              nOutputPlane / nGroup);
        }
        updateOutputFrame(
            (DenseTensor) input.narrow(dimHWD[2],
                g * nInputPlane / nGroup + 1, nInputPlane / nGroup),
            (DenseTensor) ((DenseTensor) output).narrow(dimHWD[2],
                g * nOutputPlane / nGroup + 1, nOutputPlane / nGroup),
            (DenseTensor) weightMM.select(1, g + 1),
            biasUse,
            (DenseTensor) fInput.select(1, g + 1),
            kernelW, kernelH, strideW, strideH,
            padLeft, padTop, padRight, padBottom,
            nInputPlane / nGroup, inputWidth, inputHeight,
            nOutputPlane / nGroup, outputWidth, outputHeight);
        g += 1;
      }
      //System.out.println("SpatialConv3 Time 3 : " + (System.nanoTime() - startTimef) / 1e6);

    } else {
      int batchSize = input.size(1);
      ((DenseTensor) output).resize(getOutputShape(outputHeight, outputWidth, batchSize));
      if (_1x1) {
        fInput.set(input);
        fInput.resize(getFInputShape(outputHeight, outputWidth, batchSize));
      } else {
        fInput.resize(getFInputShape(outputHeight, outputWidth, batchSize));
      }
      //System.out.println("SpatialConv Time 2 : " + (System.nanoTime() - startTimef) / 1e6);

      //TODO parallelize
      for (int i = 0; i < batchSize; i++) {
        int _i = i + 1;
        Tensor inputT = input.select(1, _i);
        Util.require(inputT.isContiguous());
        DenseTensor outputT = (DenseTensor) ((DenseTensor) output).select(1, _i);
        DenseTensor fInputT = (DenseTensor) fInput.select(1, _i);
        int g = 0;

        while (g < nGroup) {
          DenseTensor biasUse = null;
          if (withBias) {
            biasUse = (DenseTensor) bias.narrow(1, g * nOutputPlane / nGroup + 1,
                nOutputPlane / nGroup);
          }
          updateOutputFrame(
              (DenseTensor) inputT.narrow(dimHWD[2] - 1, g * nInputPlane / nGroup + 1,
                  nInputPlane / nGroup),
              (DenseTensor) outputT.narrow(dimHWD[2] - 1, g * nOutputPlane / nGroup + 1,
                  nOutputPlane / nGroup),
              (DenseTensor) weightMM.select(1, g + 1),
              biasUse,
              (DenseTensor) fInputT.select(1, g + 1),
              kernelW, kernelH, strideW, strideH,
              padLeft, padTop, padRight, padBottom,
              nInputPlane / nGroup, inputWidth, inputHeight,
              nOutputPlane / nGroup, outputWidth, outputHeight);
          g += 1;
        }

      }
      //System.out.println("SpatialConv Time 4 : " + (System.nanoTime() - startTimef) / 1e6);

    }
   // System.out.println("SpatialConv Time : " + (System.nanoTime() - startTimef) / 1e6);
    return (DenseTensor) output;
  }

  @Override
  public DenseTensor updateGradInput(DenseTensor input, DenseTensor gradOutput) {
    if (!propagateBack) {
      return (DenseTensor) gradInput;
    }

    //(ohDim, owDim, cDim)
    int[] dimOhOwC = format.getHWCDims(input.dim());
    int oh = gradOutput.size(dimOhOwC[0]);
    int ow = gradOutput.size(dimOhOwC[1]);

    int inputWidth = input.size(dimOhOwC[1]);
    int inputHeight = input.size(dimOhOwC[0]);

    //(padTop, padBottom, padLeft, padRight)
    int[] padTBLR = getPadding(inputHeight, inputWidth);

    Util.require(input.nDimension() == 3 || input.nDimension() == 4,
        "Only support 3D or 4D input");
    ((DenseTensor) gradInput).resizeAs(input);
    if (_1x1) {
      fGradInput.set((Tensor) gradInput);
      fGradInput.resizeAs(fInput);
    } else {
      fGradInput.resizeAs(fInput);
    }

    if (input.nDimension() == 3) {
      Util.require(gradOutput.isContiguous());
      int g = 0;
      while (g < nGroup) {
        updateGradInputFrame(
            (DenseTensor) ((DenseTensor) gradInput).narrow(dimOhOwC[2],
                g * nInputPlane / nGroup + 1, nInputPlane / nGroup),
            (DenseTensor) gradOutput.narrow(dimOhOwC[2], g * nOutputPlane / nGroup + 1,
                nOutputPlane / nGroup),
            (DenseTensor) weightMM.select(1, g + 1).transpose(1, 2),
            (DenseTensor) fGradInput.select(1, g + 1),
            kernelW, kernelH, strideW, strideH, padTBLR[2], padTBLR[0], padTBLR[3], padTBLR[1]);
        g += 1;
      }
    } else {
      int batchSize = input.size(1);
      for (int i = 0; i < batchSize; i++) {
        int _i = i + 1;
        DenseTensor gradInputT = (DenseTensor) ((DenseTensor) gradInput).select(1, _i);
        DenseTensor gradOutputT = (DenseTensor) gradOutput.select(1, _i);
        Util.require(gradOutputT.isContiguous());
        DenseTensor fgradInputT = (DenseTensor) fGradInput.select(1, _i);
        int g = 0;
        while (g < nGroup) {
          updateGradInputFrame(
              (DenseTensor) gradInputT.narrow(dimOhOwC[2] - 1,
                  g * nInputPlane / nGroup + 1, nInputPlane / nGroup),
              (DenseTensor) gradOutputT.narrow(dimOhOwC[2] - 1,
                  g * nOutputPlane / nGroup + 1, nOutputPlane / nGroup),
              (DenseTensor) weightMM.select(1, g + 1).transpose(1, 2),
              (DenseTensor) fgradInputT.select(1, g + 1),
              kernelW, kernelH, strideW, strideH, padTBLR[2], padTBLR[0], padTBLR[3], padTBLR[1]);
          g += 1;
        }
      }
      int i = 0;
    }

    return (DenseTensor) gradInput;
  }

  private int[] getGradWeightMMInBatchShape(int batchSize) {

    if (format instanceof NCHW) {
      return new int[]{batchSize, nGroup, nOutputPlane / nGroup,
          nInputPlane * kernelH * kernelW / nGroup};
    } else if (format instanceof NHWC) {
      return new int[]{batchSize, 1, nInputPlane * kernelH * kernelW, nOutputPlane};
    }

    return null;
  }

  @Override
  public void accGradParameters(DenseTensor input, DenseTensor gradOutput) {
    Util.require(input.nDimension() == 3 || input.nDimension() == 4,
        "Only support 3D or 4D input,"
            + "but input has ${input.nDimension()} dimension");
    Util.require(gradOutput.isContiguous());

    //(ohDim, owDim, cDim)
    int[] dimOhOwC = format.getHWCDims(input.dim());
    int oh = gradOutput.size(dimOhOwC[0]);
    int ow = gradOutput.size(dimOhOwC[1]);

    if (input.nDimension() == 3) {
      if (gradWeightMM == null) {
        gradWeightMM = (DenseTensor) gradWeight.view(weightMMShape);
      }
      int g = 0;
      while (g < nGroup) {
        DenseTensor gradBiasUse = null;
        if (withBias) {
          gradBiasUse = (DenseTensor) gradBias.narrow(1, g * nOutputPlane / nGroup + 1,
              nOutputPlane / nGroup);
        }

        accGradParametersFrame(
            (DenseTensor) gradOutput.narrow(dimOhOwC[2], g * nOutputPlane / nGroup + 1,
                nOutputPlane / nGroup),
            (DenseTensor) gradWeightMM.select(1, g + 1),
            gradBiasUse,
            (DenseTensor) fInput.select(1, g + 1),
            scaleW, scaleB);
        g += 1;
      }
    } else {
      int batchSize = input.size(1);
      if (gradWeightMMInBatch == null) {
        gradWeightMMInBatch = (DenseTensor) new DenseTensor(input.isFloat())
            .resize(getGradWeightMMInBatchShape(batchSize));
      }
      if (withBias && gradientBiasMT.nElement() == 0) {
        gradientBiasMT.resize(new int[]{batchSize, nOutputPlane});
      }
      if (ones.dim() != 1 || ones.size(1) != oh * ow) {
        if (this.isFloat) {
          ones.resize(new int[]{oh * ow}).fill(1.0f);
        } else {
          ones.resize(new int[]{oh * ow}).fill(1.0);
        }
      }

      if (onesBatch.dim() != 1 || onesBatch.size(1) != batchSize) {
        if (this.isFloat) {
          onesBatch.resize(new int[]{batchSize}).fill(1.0f);
        } else {
          onesBatch.resize(new int[]{batchSize}).fill(1.0);
        }
      }

      for (int i = 0; i < batchSize; i++) {
        int _i = i + 1;
        DenseTensor gradOutputT = (DenseTensor) gradOutput.select(1, _i);
        DenseTensor fInputT = (DenseTensor) fInput.select(1, _i);
        int g = 0;
        while (g < nGroup) {
          DenseTensor gradientBiasMTUse = null;
          if (withBias) {
            gradientBiasMTUse = (DenseTensor) gradientBiasMT.select(1, _i)
                .narrow(1, g * nOutputPlane / nGroup + 1,
                    nOutputPlane / nGroup);
          }
          calcGradParametersFrame(
              (DenseTensor) gradOutputT.narrow(dimOhOwC[2] - 1,
                  g * nOutputPlane / nGroup + 1, nOutputPlane / nGroup),
              (DenseTensor) gradWeightMMInBatch.select(1, _i).select(1, g + 1),
              gradientBiasMTUse,
              (DenseTensor) fInputT.select(1, g + 1),
              scaleW, scaleB);
          g += 1;
        }

      }
      DenseTensor gradView = (DenseTensor) gradWeightMMInBatch.view(new int[]{batchSize,
          nOutputPlane * nInputPlane * kernelH * kernelW / nGroup}).t();
      DenseTensor grad = (DenseTensor) gradWeight
          .view(new int[]{nOutputPlane * nInputPlane * kernelH * kernelW / nGroup});
      if (this.isFloat) {
        grad.addmv(1.0f, 1.0f, gradView, onesBatch);
        if (withBias) {
          gradBias.addmv(1.0f, 1.0f, gradientBiasMT.t(), onesBatch);
        }

        if (null != wRegularizer) {
          wRegularizer.accRegularization(weight, gradWeight, (float) scaleW);
        }
        if (withBias && null != bRegularizer) {
          bRegularizer.accRegularization(bias, gradBias, (float) scaleB);
        }
      } else {
        grad.addmv(1.0, 1.0, gradView, onesBatch);
        if (withBias) {
          gradBias.addmv(1.0, 1.0, gradientBiasMT.t(), onesBatch);
        }

        if (null != wRegularizer) {
          wRegularizer.accRegularization(weight, gradWeight, scaleW);
        }
        if (withBias && null != bRegularizer) {
          bRegularizer.accRegularization(bias, gradBias, scaleB);
        }
      }
    }
  }

  @Override
  public TensorArrayPair parameters() {
    if (withBias) {
      return new TensorArrayPair(new DenseTensor[]{this.weight, this.bias},
          new DenseTensor[]{this.gradWeight, this.gradBias});
    } else {
      return new TensorArrayPair(new DenseTensor[]{this.weight},
          new DenseTensor[]{this.gradWeight});
    }
  }

  @Override
  public boolean equals(Object obj) {
    if (!super.equals(obj)) {
      return false;
    }

    if (!(obj instanceof SpatialConvolution)) {
      return false;
    }

    SpatialConvolution other = (SpatialConvolution) obj;

    if (this == other) {
      return true;
    }

    return nInputPlane == other.nInputPlane
        && nOutputPlane == other.nOutputPlane
        && kernelW == other.kernelW
        && kernelH == other.kernelH
        && strideW == other.strideW
        && strideH == other.strideH
        && padW == other.padW
        && padH == other.padH
        && nGroup == other.nGroup
        && propagateBack == other.propagateBack
        && weight == other.weight
        && bias == other.bias
        && gradWeight == other.gradWeight
        && gradBias == other.gradBias;
  }

  @Override
  public int hashCode() {
    int seed = 37;
    int hash = super.hashCode();
    hash = hash * seed + nInputPlane;
    hash = hash * seed + nOutputPlane;
    hash = hash * seed + kernelW;
    hash = hash * seed + kernelH;
    hash = hash * seed + strideW;
    hash = hash * seed + strideH;
    hash = hash * seed + padW;
    hash = hash * seed + padH;
    hash = hash * seed + weight.hashCode();
    if (withBias) hash = hash * seed + bias.hashCode();
    hash = hash * seed + gradWeight.hashCode();
    if (withBias) hash = hash * seed + gradBias.hashCode();
    return hash;
  }

  @Override
  public SpatialConvolution clearState() {
    super.clearState();
    fInput.set();
    fGradInput.set();
    ones.set();
    onesBatch.set();
    if (withBias) {
      onesBias.set();
      gradientBiasMT.set();
    }
    return this;
  }

  @Override
  public String toString() {
    return String.format("%s (%d -> %d, %d x %d, %d, %d, %d, %d)", getPrintName(),
        nInputPlane, nOutputPlane, kernelW, kernelH, strideW, strideH, padW, padH);
  }

  protected void updateOutputFrame(DenseTensor input, DenseTensor output, DenseTensor weight,
                                   DenseTensor bias, DenseTensor fInput, int kW, int kH, int dW,
                                   int dH, int padLeft, int padTop, int padRight, int padBottom,
                                   int nInputPlane, int inputWidth, int inputHeight,
                                   int nOutputPlane, int outputWidth, int outputHeight) {
    long startTimef = System.nanoTime();
    if (this.isFloat) {
      if (format instanceof NCHW) {
        long startTime2d = System.nanoTime();

        DenseTensor output2d = (DenseTensor) output.view(new int[]{nOutputPlane,
            outputHeight * outputWidth});
       // System.out.println("Update Frame output2d: " + (System.nanoTime() - startTime2d) / 1e6);
        if (!_1x1) {
          long before = System.nanoTime();
          NNPrimitive.im2colFloat(fInput,
              input, kW, kH, dW, dH,
              padLeft, padTop, padRight, padBottom,
              outputWidth, outputHeight);
          im2colTime += System.nanoTime() - before;
        }
        output2d.addmm(0.0f, output2d, 1.0f, weight, fInput);
        if (withBias) output2d.addr(1.0f, bias, onesBias);
      } else if (format instanceof NHWC) {
        //System.out.println(" ##################### NHWC ############Update Frame : ");
        DenseTensor output2d = (DenseTensor) output
            .view(new int[]{outputHeight * outputWidth, nOutputPlane});
        if (!_1x1) {
          long before = System.nanoTime();
          NNPrimitive.im2colFloatNHWC(fInput,
              input, kW, kH, dW, dH,
              padLeft, padTop, padRight, padBottom,
              outputWidth, outputHeight);
          im2colTime += System.nanoTime() - before;
        }
        output2d.addmm(0.0f, output2d, 1.0f, fInput, weight);
        if (withBias) output2d.addr(1.0f, onesBias, bias);
      }
    } else {
      //System.out.println(" ##################### Double ############Update Frame : ");
      if (format instanceof NCHW) {
        DenseTensor output2d = (DenseTensor) output.view(new int[]{nOutputPlane,
            outputHeight * outputWidth});
        if (!_1x1) {
          long before = System.nanoTime();
          NNPrimitive.im2colDouble(fInput,
              input, kW, kH, dW, dH,
              padLeft, padTop, padRight, padBottom,
              outputWidth, outputHeight);
          im2colTime += System.nanoTime() - before;
        }
        output2d.addmm(0.0, output2d, 1.0, weight, fInput);
        if (withBias) output2d.addr(1.0, bias, onesBias);
      } else if (format instanceof NHWC) {
        DenseTensor output2d = (DenseTensor) output
            .view(new int[]{outputHeight * outputWidth, nOutputPlane});
        if (!_1x1) {
          long before = System.nanoTime();
          NNPrimitive.im2colDoubleNHWC(fInput,
              input, kW, kH, dW, dH,
              padLeft, padTop, padRight, padBottom,
              outputWidth, outputHeight);
          im2colTime += System.nanoTime() - before;
        }
        output2d.addmm(0.0, output2d, 1.0, fInput, weight);
        if (withBias) output2d.addr(1.0, onesBias, bias);
      }
    }
  //  System.out.println("Update Frame : " + (System.nanoTime() - startTimef) / 1e6);

  }

  protected void updateGradInputFrame(
      DenseTensor gradInput, DenseTensor gradOutput,
      DenseTensor weight, DenseTensor fgradInput, int kW, int kH, int dW, int dH, int
          padLeft, int padTop, int padRight, int padBottom) {

    DenseTensor gradOutDouble = gradOutput;
    DenseTensor fGradInDouble = fgradInput;
    DenseTensor weightDouble = weight;
    DenseTensor gradInputDouble = gradInput;

    if (this.isFloat) {
      if (format instanceof NCHW) {
        int channel = gradOutDouble.size(1);
        int oh = gradOutDouble.size(2);
        int ow = gradOutDouble.size(3);
        Tensor gradOutput2d = gradOutDouble.view(new int[]{channel, oh * ow});
        fGradInDouble.addmm(0.0f, fGradInDouble, 1.0f, weightDouble, gradOutput2d);
        if (!_1x1) {
          gradInputDouble.zero();
          long before = System.nanoTime();
          NNPrimitive.col2imFloat(fGradInDouble,
              gradInputDouble, kW, kH, dW, dH,
              padLeft, padTop, padRight, padBottom,
              gradOutput.size(3), gradOutput.size(2));
          col2imTime += System.nanoTime() - before;
        }
      } else if (format instanceof NHWC) {
        int channel = gradOutDouble.size(3);
        int oh = gradOutDouble.size(1);
        int ow = gradOutDouble.size(2);
        Tensor gradOutput2d = gradOutDouble.view(new int[]{oh * ow, channel});
        fGradInDouble.addmm(0.0f, fGradInDouble, 1.0f, gradOutput2d, weightDouble);
        if (!_1x1) {
          gradInputDouble.zero();
          long before = System.nanoTime();
          NNPrimitive.col2imFloatNHWC(fGradInDouble,
              gradInputDouble, kW, kH, dW, dH,
              padLeft, padTop, padRight, padBottom,
              gradOutput.size(2), gradOutput.size(1));
          col2imTime += System.nanoTime() - before;
        }
      }
    } else {
      if (format instanceof NCHW) {
        int channel = gradOutDouble.size(1);
        int oh = gradOutDouble.size(2);
        int ow = gradOutDouble.size(3);
        Tensor gradOutput2d = gradOutDouble.view(new int[]{channel, oh * ow});
        fGradInDouble.addmm(0.0, fGradInDouble, 1.0, weightDouble, gradOutput2d);
        if (!_1x1) {
          gradInputDouble.zero();
          long before = System.nanoTime();
          NNPrimitive.col2imDouble(fGradInDouble,
              gradInputDouble, kW, kH, dW, dH,
              padLeft, padTop, padRight, padBottom,
              gradOutput.size(3), gradOutput.size(2));
          col2imTime += System.nanoTime() - before;
        }
      } else if (format instanceof NHWC) {
        int channel = gradOutDouble.size(3);
        int oh = gradOutDouble.size(1);
        int ow = gradOutDouble.size(2);
        Tensor gradOutput2d = gradOutDouble.view(new int[]{oh * ow, channel});
        fGradInDouble.addmm(0.0, fGradInDouble, 1.0, gradOutput2d, weightDouble);
        if (!_1x1) {
          gradInputDouble.zero();
          long before = System.nanoTime();
          NNPrimitive.col2imDoubleNHWC(fGradInDouble,
              gradInputDouble, kW, kH, dW, dH,
              padLeft, padTop, padRight, padBottom,
              gradOutput.size(2), gradOutput.size(1));
          col2imTime += System.nanoTime() - before;
        }
      }
    }

  }

  protected void accGradParametersFrame(DenseTensor gradOutput, DenseTensor gradWeight,
                                        DenseTensor gradBias, DenseTensor fInput,
                                        double scaleW, double scaleB) {

    DenseTensor gradODouble = gradOutput;
    DenseTensor gradWDouble = gradWeight;
    DenseTensor fIDouble = fInput;
    double sWDouble = scaleW;
    double sBDouble = scaleB;
    DenseTensor gradBDouble = gradBias;

    if (this.isFloat) {
      float sWDoublef = (float) scaleW;
      float sBDoublef = (float) scaleB;
      if (format instanceof NCHW) {
        int outChannel = gradOutput.size(1);
        int outSize = gradOutput.size(2) * gradOutput.size(3);
        Tensor gradOutput2d = gradODouble.view(new int[]{outChannel, outSize});
        if (sWDoublef != 0) {
          gradWDouble.addmm(1.0f, gradWDouble, sWDoublef, gradOutput2d, fIDouble.t());
        }
        if (withBias && sBDoublef != 0) {
          int i = 0;
          while (i < gradBias.size(1)) {
            float sum = 0.0f;
            float[] data = gradOutput2d.storage().toFloatArray();
            int offset = gradOutput2d.storageOffset() - 1 + i * gradOutput2d.stride(1);
            int k = 0;
            while (k < gradOutput2d.size(2)) {
              sum += data[k + offset];
              k += 1;
            }
            gradBDouble.setValue(i + 1, gradBDouble.valueAtf(i + 1) + (sBDoublef * sum));
            i += 1;
          }
        }
      } else if (format instanceof NHWC) {
        int outChannel = gradOutput.size(3);
        int outSize = gradOutput.size(1) * gradOutput.size(2);
        Tensor gradOutput2d = gradODouble.view(new int[]{outSize, outChannel});

        if (sWDoublef != 0) {
          gradWDouble.addmm(1.0f, gradWDouble, sWDoublef, fIDouble.t(), gradOutput2d);
        }

        if (sBDoublef != 0) {
          int i = 0;
          float[] gradData = gradOutput2d.storage().toFloatArray();
          float[] biasData = gradBDouble.storage().toFloatArray();
          int biasOffset = gradBDouble.storageOffset() - 1;

          while (i < gradODouble.size(1)) {
            int gradOffset = gradOutput2d.storageOffset() - 1 + i * gradOutput2d.stride(1);
            int j = 0;
            while (j < gradOutput2d.size(2)) {
              biasData[biasOffset + j] += gradData[gradOffset + j];
              j = j + 1;
            }
            i = i + 1;
          }
        }
      }
    } else {
      if (format instanceof NCHW) {
        int outChannel = gradOutput.size(1);
        int outSize = gradOutput.size(2) * gradOutput.size(3);
        Tensor gradOutput2d = gradODouble.view(new int[]{outChannel, outSize});
        if (sWDouble != 0) {
          gradWDouble.addmm(1.0, gradWDouble, sWDouble, gradOutput2d, fIDouble.t());
        }
        if (withBias && sBDouble != 0) {
          int i = 0;
          while (i < gradBias.size(1)) {
            double sum = 0.0;
            double[] data = gradOutput2d.storage().toDoubleArray();
            int offset = gradOutput2d.storageOffset() - 1 + i * gradOutput2d.stride(1);
            int k = 0;
            while (k < gradOutput2d.size(2)) {
              sum += data[k + offset];
              k += 1;
            }
            gradBDouble.setValue(i + 1, gradBDouble.valueAt(i + 1) + (sBDouble * sum));
            i += 1;
          }
        }
      } else if (format instanceof NHWC) {
        int outChannel = gradOutput.size(3);
        int outSize = gradOutput.size(1) * gradOutput.size(2);
        Tensor gradOutput2d = gradODouble.view(new int[]{outSize, outChannel});

        if (sWDouble != 0) {
          gradWDouble.addmm(1.0, gradWDouble, sWDouble, fIDouble.t(), gradOutput2d);
        }

        if (sBDouble != 0) {
          int i = 0;
          double[] gradData = gradOutput2d.storage().toDoubleArray();
          double[] biasData = gradBDouble.storage().toDoubleArray();
          int biasOffset = gradBDouble.storageOffset() - 1;

          while (i < gradODouble.size(1)) {
            int gradOffset = gradOutput2d.storageOffset() - 1 + i * gradOutput2d.stride(1);
            int j = 0;
            while (j < gradOutput2d.size(2)) {
              biasData[biasOffset + j] += gradData[gradOffset + j];
              j = j + 1;
            }
            i = i + 1;
          }
        }
      }
    }
  }

  protected void calcGradParametersFrame(DenseTensor gradOutput, DenseTensor gradWeight,
                                         DenseTensor gradBias,
                                         DenseTensor fInput, double scaleW, double scaleB) {
    DenseTensor gradODouble = gradOutput;
    DenseTensor gradWDouble = gradWeight;
    double sWDouble = scaleW;
    double sBDouble = scaleB;
    DenseTensor fIDouble = fInput;
    DenseTensor gradBDouble = gradBias;
    DenseTensor onesDouble = ones;

    if (this.isFloat) {
      float sWDoublef = (float) scaleW;
      float sBDoublef = (float) scaleB;
      if (format instanceof NCHW) {
        int channel = gradODouble.size(1);
        int oh = gradODouble.size(2);
        int ow = gradODouble.size(3);
        Tensor gradOutput2d = gradODouble.view(new int[]{channel, oh * ow});

        if (scaleW != 0) {
          gradWDouble.addmm(0.0f, gradWDouble, sWDoublef, gradOutput2d, fIDouble.t());
        }

        if (withBias && scaleB != 0) {
          gradBDouble.addmv(0.0f, sBDoublef, gradOutput2d, onesDouble);
        }
      } else if (format instanceof NHWC) {
        int channel = gradODouble.size(3);
        int oh = gradODouble.size(1);
        int ow = gradODouble.size(2);
        Tensor gradOutput2d = gradODouble.view(new int[]{oh * ow, channel});

        if (scaleW != 0) {
          gradWDouble.addmm(0.0f, gradWDouble, sWDoublef, fIDouble.t(), gradOutput2d);
        }

        if (withBias && scaleB != 0) {
          gradBDouble.addmv(0.0f, sBDoublef, gradOutput2d.t(), onesDouble);
        }
      }
    } else {
      if (format instanceof NCHW) {
        int channel = gradODouble.size(1);
        int oh = gradODouble.size(2);
        int ow = gradODouble.size(3);
        Tensor gradOutput2d = gradODouble.view(new int[]{channel, oh * ow});

        if (scaleW != 0) {
          gradWDouble.addmm(0.0, gradWDouble, sWDouble, gradOutput2d, fIDouble.t());
        }

        if (withBias && scaleB != 0) {
          gradBDouble.addmv(0.0, sBDouble, gradOutput2d, onesDouble);
        }
      } else if (format instanceof NHWC) {
        int channel = gradODouble.size(3);
        int oh = gradODouble.size(1);
        int ow = gradODouble.size(2);
        Tensor gradOutput2d = gradODouble.view(new int[]{oh * ow, channel});

        if (scaleW != 0) {
          gradWDouble.addmm(0.0, gradWDouble, sWDouble, fIDouble.t(), gradOutput2d);
        }

        if (withBias && scaleB != 0) {
          gradBDouble.addmv(0.0, sBDouble, gradOutput2d.t(), onesDouble);
        }
      }
    }
  }

  @Override
  public Initializable setInitMethod(InitializationMethod weightMethod,
                                     InitializationMethod biasMethod) {
    if (weightMethod != null) {
      this.weightInitMethod = weightMethod;
    }

    if (biasMethod != null) {
      this.biasInitMethod = biasMethod;
    }
    reset();
    return this;
  }
}
