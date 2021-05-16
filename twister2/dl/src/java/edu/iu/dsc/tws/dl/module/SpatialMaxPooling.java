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

import edu.iu.dsc.tws.dl.data.format.DataFormat;
import edu.iu.dsc.tws.dl.data.format.NCHW;
import edu.iu.dsc.tws.dl.data.format.NHWC;
import edu.iu.dsc.tws.dl.data.tensor.DenseTensor;
import edu.iu.dsc.tws.dl.graph.TensorModule;
import edu.iu.dsc.tws.dl.utils.ErrorConstants;
import edu.iu.dsc.tws.dl.utils.NNPrimitive;
import edu.iu.dsc.tws.dl.utils.Util;
import edu.iu.dsc.tws.dl.utils.pair.TensorArrayPair;

/**
 * Applies 2D max-pooling operation in kWxkH regions by step size dWxdH steps.
 * The number of output features is equal to the number of input planes.
 * If the input image is a 3D tensor nInputPlane x height x width,
 * the output image size will be nOutputPlane x oheight x owidth where
 * owidth  = op((width  + 2*padW - kW) / dW + 1)
 * oheight = op((height + 2*padH - kH) / dH + 1)
 * op is a rounding operator. By default, it is floor.
 * It can be changed by calling :ceil() or :floor() methods.
 * <p>
 * When padW and padH are both -1, we use a padding algorithm similar to the "SAME"
 * padding of tensorflow. That is
 * <p>
 * outHeight = Math.ceil(inHeight.toFloat/strideH.toFloat)
 * outWidth = Math.ceil(inWidth.toFloat/strideW.toFloat)
 * <p>
 * padAlongHeight = Math.max(0, (outHeight - 1) * strideH + kernelH - inHeight)
 * padAlongWidth = Math.max(0, (outWidth - 1) * strideW + kernelW - inWidth)
 * <p>
 * padTop = padAlongHeight / 2
 * padLeft = padAlongWidth / 2
 * <p>
 * kW              kernel width
 * kH              kernel height
 * dW              step size in width
 * dH              step size in height
 * padW            padding in width
 * padH            padding in height
 * format          DataFormat.NCHW or DataFormat.NHWC, indicating the input
 * data format
 */
public class SpatialMaxPooling extends TensorModule<DenseTensor> {

  private int kW;
  private int kH;
  private int dW;
  private int dH;
  private int padW = 0;
  private int padH = 0;
  private DataFormat format = new NCHW();

  private boolean ceilMode;
  private DenseTensor indices = new DenseTensor(false);

  public SpatialMaxPooling(int kW, int kH) {
    this(kW, kH, 1, 1, 0, 0, new NCHW());
  }

  public SpatialMaxPooling(int kW, int kH, int dW, int dH) {
    this(kW, kH, dW, dH, 0, 0, new NCHW());
  }

  public SpatialMaxPooling(int kW, int kH, int dW, int dH, int padW, int padH, DataFormat format) {
    this.kW = kW;
    this.kH = kH;
    this.dW = dW;
    this.dH = dH;
    this.padW = padW;
    this.padH = padH;
    this.format = format;
    this.output = new DenseTensor(this.isFloat);
    this.gradInput = new DenseTensor(this.isFloat);
  }

  @Override
  public void toFloat() {
    super.toFloat();
    indices = new DenseTensor(true);
    this.output = new DenseTensor(this.isFloat);
    this.gradInput = new DenseTensor(this.isFloat);
  }

  /**
   * set ceil mode
   *
   * @return this
   */
  public SpatialMaxPooling ceil() {
    ceilMode = true;
    return this;
  }

  /**
   * set floor mode
   *
   * @return this
   */
  public SpatialMaxPooling floor() {
    ceilMode = false;
    return this;
  }

  @Override
  public DenseTensor updateOutput(DenseTensor input) {
    Util.require(input.dim() == 3 || input.dim() == 4,
        "SpatialMaxPooling: " + ErrorConstants.constrainInputAs3DOrBatch);

    //(dimh, dimw, dimc)
    int[] dimHWC = format.getHWCDims(input.dim());

    int nInputPlane = input.size(dimHWC[2]);
    int inputHeight = input.size(dimHWC[0]);
    int inputWidth = input.size(dimHWC[1]);

    int[] sizes;
    if (padW == -1 && padH == -1) {
      // no ceil/floor mode in SAME padding
      sizes = Util.getSAMEOutSizeAndPadding(inputHeight, inputWidth, dH, dW, kH, kW);
    } else {
      Util.require(inputWidth >= kW - padW && inputHeight >= kH - padH,
          "input smaller than kernel size"
              + "input size(${input.size(dimw)},${input.size(dimh)})"
              + "kernel size(${kW-padW},${kH-padH})");
      Util.require(kW / 2 >= padW && kH / 2 >= padH,
          "pad should be smaller than half of kernel size"
              + "pad size($padW,$padH) kernel size($kW, $kH)");

      sizes = Util.getOutSizeAndPadding(inputHeight, inputWidth, dH, dW, kH,
          kW, padH, padW, ceilMode);
    }

    int padTop = sizes[0];
    int padBottom = sizes[1];
    int padLeft = sizes[2];
    int padRight = sizes[3];
    int oHeight = sizes[4];
    int oWidth = sizes[5];

    if (ceilMode && padW == 0 && (inputWidth - kW) % dW == 0) {
      ceilMode = false; // The ceil mode is not needed.
    }

    if (this.isFloat) {
      if (input.dim() == 3) {

        if (format instanceof NCHW) {
          ((DenseTensor) output).resize(new int[]{nInputPlane, oHeight, oWidth});
          /* indices will contain the locations for each output point */
          indices.resize(new int[]{nInputPlane, oHeight, oWidth});

          NNPrimitive.maxPoolingForwardFloat(
              input,
              (DenseTensor) output,
              indices,
              oWidth, oHeight, kW, kH, dW, dH, padLeft, padTop);

        } else if (format instanceof NHWC) {
          ((DenseTensor) output).resize(new int[]{oHeight, oWidth, nInputPlane});
          /* indices will contain the locations for each output point */
          indices.resize(new int[]{oHeight, oWidth, nInputPlane});

          NNPrimitive.maxPoolingForwardFloatNHWC(
              input,
              (DenseTensor) output,
              indices,
              oWidth, oHeight, kW, kH, dW, dH, padLeft, padTop);

        }
      } else {
        int nbatch = input.size(1);
        if (format instanceof NCHW) {
          ((DenseTensor) output).resize(new int[]{nbatch, nInputPlane, oHeight, oWidth});
          indices.resize(new int[]{nbatch, nInputPlane, oHeight, oWidth});
          for (int i = 1; i <= nbatch; i++) {
            DenseTensor curInput = (DenseTensor) input.apply(i);
            DenseTensor curOutput = (DenseTensor) ((DenseTensor) output).apply(i);
            DenseTensor curIndices = (DenseTensor) indices.apply(i);
            NNPrimitive.maxPoolingForwardFloat(
                curInput,
                curOutput,
                curIndices,
                oWidth, oHeight,
                kW, kH, dW, dH, padLeft, padTop);
          }
        } else if (format instanceof NHWC) {
          ((DenseTensor) output).resize(new int[]{nbatch, oHeight, oWidth, nInputPlane});
          indices.resize(new int[]{nbatch, oHeight, oWidth, nInputPlane});

          for (int i = 1; i <= nbatch; i++) {
            DenseTensor curInput = (DenseTensor) input.apply(i);
            DenseTensor curOutput = (DenseTensor) ((DenseTensor) output).apply(i);
            DenseTensor curIndices = (DenseTensor) indices.apply(i);
            NNPrimitive.maxPoolingForwardFloatNHWC(
                curInput,
                curOutput,
                curIndices,
                oWidth, oHeight,
                kW, kH, dW, dH, padLeft, padTop);
          }
        }
      }
    } else {
      if (input.dim() == 3) {

        if (format instanceof NCHW) {
          ((DenseTensor) output).resize(new int[]{nInputPlane, oHeight, oWidth});
          /* indices will contain the locations for each output point */
          indices.resize(new int[]{nInputPlane, oHeight, oWidth});

          NNPrimitive.maxPoolingForwardDouble(
              input,
              (DenseTensor) output,
              indices,
              oWidth, oHeight, kW, kH, dW, dH, padLeft, padTop);

        } else if (format instanceof NHWC) {
          ((DenseTensor) output).resize(new int[]{oHeight, oWidth, nInputPlane});
          /* indices will contain the locations for each output point */
          indices.resize(new int[]{oHeight, oWidth, nInputPlane});

          NNPrimitive.maxPoolingForwardDoubleNHWC(
              input,
              (DenseTensor) output,
              indices,
              oWidth, oHeight, kW, kH, dW, dH, padLeft, padTop);

        }
      } else {
        int nbatch = input.size(1);
        if (format instanceof NCHW) {
          ((DenseTensor) output).resize(new int[]{nbatch, nInputPlane, oHeight, oWidth});
          indices.resize(new int[]{nbatch, nInputPlane, oHeight, oWidth});
          for (int i = 1; i <= nbatch; i++) {
            DenseTensor curInput = (DenseTensor) input.apply(i);
            DenseTensor curOutput = (DenseTensor) ((DenseTensor) output).apply(i);
            DenseTensor curIndices = (DenseTensor) indices.apply(i);
            NNPrimitive.maxPoolingForwardDouble(
                curInput,
                curOutput,
                curIndices,
                oWidth, oHeight,
                kW, kH, dW, dH, padLeft, padTop);
          }
        } else if (format instanceof NHWC) {
          ((DenseTensor) output).resize(new int[]{nbatch, oHeight, oWidth, nInputPlane});
          indices.resize(new int[]{nbatch, oHeight, oWidth, nInputPlane});

          for (int i = 1; i <= nbatch; i++) {
            DenseTensor curInput = (DenseTensor) input.apply(i);
            DenseTensor curOutput = (DenseTensor) ((DenseTensor) output).apply(i);
            DenseTensor curIndices = (DenseTensor) indices.apply(i);
            NNPrimitive.maxPoolingForwardDoubleNHWC(
                curInput,
                curOutput,
                curIndices,
                oWidth, oHeight,
                kW, kH, dW, dH, padLeft, padTop);
          }
        }
      }
    }
    return (DenseTensor) output;
  }

  @Override
  public DenseTensor updateGradInput(DenseTensor input, DenseTensor gradOutput) {

    //(dimh, dimw, dimc)
    int[] dimHWC = format.getHWCDims(input.dim());

    int oHeight = gradOutput.size(dimHWC[0]);
    int oWidth = gradOutput.size(dimHWC[1]);
    ((DenseTensor) gradInput).resizeAs(input);
    ((DenseTensor) gradInput).zero();

    if (this.isFloat) {
      if (input.dim() == 3) {

        if (format instanceof NCHW) {
          NNPrimitive.maxPoolingBackwardFloat(
              (DenseTensor) gradInput,
              gradOutput,
              indices,
              oWidth, oHeight);
        } else if (format instanceof NHWC) {
          NNPrimitive.maxPoolingBackwardFloatNHWC(
              (DenseTensor) gradInput,
              gradOutput,
              indices,
              oWidth, oHeight);
        }
      } else {
        int nbatch = input.size(1);

        if (format instanceof NCHW) {
          for (int k = 1; k <= nbatch; k++) {
            DenseTensor curGradInput = (DenseTensor) ((DenseTensor) gradInput).apply(k);
            DenseTensor curGradOutput = (DenseTensor) gradOutput.apply(k);
            DenseTensor curIndices = (DenseTensor) indices.apply(k);
            NNPrimitive.maxPoolingBackwardFloat(
                curGradInput,
                curGradOutput,
                curIndices,
                oWidth, oHeight);
          }
        } else if (format instanceof NHWC) {
          for (int k = 1; k <= nbatch; k++) {
            DenseTensor curGradInput = (DenseTensor) ((DenseTensor) gradInput).apply(k);
            DenseTensor curGradOutput = (DenseTensor) gradOutput.apply(k);
            DenseTensor curIndices = (DenseTensor) indices.apply(k);
            NNPrimitive.maxPoolingBackwardFloatNHWC(
                curGradInput,
                curGradOutput,
                curIndices,
                oWidth, oHeight);
          }
        }
      }
    } else {
      if (input.dim() == 3) {

        if (format instanceof NCHW) {
          NNPrimitive.maxPoolingBackwardDouble(
              (DenseTensor) gradInput,
              gradOutput,
              indices,
              oWidth, oHeight);
        } else if (format instanceof NHWC) {
          NNPrimitive.maxPoolingBackwardDoubleNHWC(
              (DenseTensor) gradInput,
              gradOutput,
              indices,
              oWidth, oHeight);
        }
      } else {
        int nbatch = input.size(1);

        if (format instanceof NCHW) {
          for (int k = 1; k <= nbatch; k++) {
            DenseTensor curGradInput = (DenseTensor) ((DenseTensor) gradInput).apply(k);
            DenseTensor curGradOutput = (DenseTensor) gradOutput.apply(k);
            DenseTensor curIndices = (DenseTensor) indices.apply(k);
            NNPrimitive.maxPoolingBackwardDouble(
                curGradInput,
                curGradOutput,
                curIndices,
                oWidth, oHeight);
          }
        } else if (format instanceof NHWC) {
          for (int k = 1; k <= nbatch; k++) {
            DenseTensor curGradInput = (DenseTensor) ((DenseTensor) gradInput).apply(k);
            DenseTensor curGradOutput = (DenseTensor) gradOutput.apply(k);
            DenseTensor curIndices = (DenseTensor) indices.apply(k);
            NNPrimitive.maxPoolingBackwardDoubleNHWC(
                curGradInput,
                curGradOutput,
                curIndices,
                oWidth, oHeight);
          }
        }
      }
    }

    return (DenseTensor) gradInput;
  }

  @Override
  public AbstractModule clearState() {
    super.clearState();
    indices.set();
    return this;
  }

  @Override
  public String toString() {
    return String.format("%s (%d, %d, %d, %d, %d, %d)", getPrintName(), kW, kH, dW, dH, padW, padH);
  }

  @Override
  public int hashCode() {
    int seed = 37;
    int hash = super.hashCode();
    hash = hash * seed + kW;
    hash = hash * seed + kH;
    hash = hash * seed + dW;
    hash = hash * seed + dH;
    hash = hash * seed + padW;
    hash = hash * seed + padH;
    hash = hash * seed + ((Boolean) ceilMode).hashCode();
    hash = hash * seed + indices.hashCode();

    return hash;
  }

  @Override
  public boolean equals(Object other) {
    if (!super.equals(other)) {
      return false;
    }

    if (!(other instanceof SpatialMaxPooling)) {
      return false;
    }

    SpatialMaxPooling spmp = (SpatialMaxPooling) other;
    if (this == spmp) {
      return true;
    }

    return kW == spmp.kW
        && kH == spmp.kH
        && dW == spmp.dW
        && dH == spmp.dH
        && padW == spmp.padW
        && padH == spmp.padH
        && ceilMode == spmp.ceilMode
        && indices == spmp.indices;
  }

  @Override
  public TensorArrayPair parameters() {
    return null;
  }

  @Override
  public void reset() {

  }
}
