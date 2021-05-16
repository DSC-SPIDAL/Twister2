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
package edu.iu.dsc.tws.dl.utils;

import java.util.Arrays;

import edu.iu.dsc.tws.dl.data.storage.ArrayDoubleStorage;
import edu.iu.dsc.tws.dl.data.storage.ArrayFloatStorage;
import edu.iu.dsc.tws.dl.data.storage.ArrayStorage;
import edu.iu.dsc.tws.dl.module.mkldnn.MemoryData;

public final class Util {

  private Util() {
  }

  public static ArrayStorage buildStorage(int size, boolean isFloat) {
    if (isFloat) {
      return new ArrayFloatStorage(size);
    } else {
      return new ArrayDoubleStorage(size);
    }
  }

  public static void require(boolean satisfied, String message) {


    if (!satisfied) {
      throw new IllegalStateException(message);
    }
  }

  public static void require(boolean satisfied) {
    if (!satisfied) {
      throw new IllegalStateException("Requirement not met");
    }
  }

  public static int getHashCode(Object o) {
    if (o == null) {
      return 0;
    } else {
      return o.hashCode();
    }
  }

  public static int[] getSAMEOutSizeAndPadding(int inputHeight, int inputWidth, int dH, int dW,
                                               int kH, int kW) {
    return getSAMEOutSizeAndPadding(inputHeight, inputWidth, dH, dW, kH, kW, -1, -1, -1);
  }

  /**
   * Default values inputDepth = -1, dT = -1, kT = -1.
   *
   * @return Array(padTop, padBottom, padLeft, padRight, outputHeight, outputWidth)
   * or Array(padFront, padBackward, padTop, padBottom, padLeft, padRight,
   * outputDepth, outputHeight, outputWidth)
   */
  public static int[] getSAMEOutSizeAndPadding(int inputHeight, int inputWidth, int dH, int dW,
                                               int kH, int kW, int inputDepth, int dT, int kT) {

    int oW = (int) Math.ceil((double) inputWidth / (double) dW);
    int oH = (int) Math.ceil((double) inputHeight / (double) dH);
    int padAlongWidth = Math.max(0, (oW - 1) * dW + kW - inputWidth);
    int padAlongHeight = Math.max(0, (oH - 1) * dH + kH - inputHeight);
    if (inputDepth != -1) {
      require(dT > 0 && kT > 0, "kernel size and strideSize cannot be smaller than 0");
      int oT = (int) Math.ceil((double) inputDepth / (double) dT);
      int padAlongDepth = Math.max(0, (oT - 1) * dT + kT - inputDepth);
      return new int[]{padAlongDepth / 2, padAlongDepth - padAlongDepth / 2, padAlongHeight / 2,
          padAlongHeight - padAlongHeight / 2, padAlongWidth / 2, padAlongWidth - padAlongWidth / 2,
          oT, oH, oW};
    }
    return new int[]{padAlongHeight / 2, padAlongHeight - padAlongHeight / 2,
        padAlongWidth / 2, padAlongWidth - padAlongWidth / 2,
        oH, oW};
  }

  public static int[] getOutSizeAndPadding(
      int inputHeight,
      int inputWidth,
      int dH,
      int dW,
      int kH,
      int kW,
      int padH,
      int padW,
      boolean ceilMode) {
    return getOutSizeAndPadding(inputHeight, inputWidth, dH, dW, kH, kW, padH, padW, ceilMode,
        1, 1, -1, -1, -1, 0, 1);
  }

  /**
   * Default values, dilationHeight = 1, dilationWidth = 1,inputdepth = -1,
   * dt = -1,kt = -1, padt = 0, dilationDepth = 1
   *
   * @return Array(padLeft, padRight, padTop, padBottom, outputHeight, outputWidth)
   * or Array(padFront, padBack, padLeft, padRight, padTop, padBottom,
   * outputDepth, outputHeight, outputWidth)
   */
  public static int[] getOutSizeAndPadding(
      int inputHeight,
      int inputWidth,
      int dH,
      int dW,
      int kH,
      int kW,
      int padH,
      int padW,
      boolean ceilMode,
      int dilationHeight,
      int dilationWidth,
      int inputdepth,
      int dt,
      int kt,
      int padt,
      int dilationDepth) {
    int oheight = 0;
    int owidth = 0;
    int odepth = 0;

    int dilationKernelHeight = dilationHeight * (kH - 1) + 1;
    int dilationKernelWidth = dilationWidth * (kW - 1) + 1;
    int dilationKernelDepth;
    if (inputdepth > 0) {
      dilationKernelDepth = dilationDepth * (kt - 1) + 1;
    } else {
      dilationKernelDepth = kt;
    }

    if (ceilMode) {
      oheight = (int) (Math.ceil(1.0 * (inputHeight - dilationKernelHeight + 2 * padH) / dH) + 1);
      owidth = (int) Math.ceil(1.0 * (inputWidth - dilationKernelWidth + 2 * padW) / dW) + 1;
      if (inputdepth > 0) {
        require(dt > 0 && kt > 0 && padt >= 0,
            "kernel size, stride size, padding size cannot be smaller than 0");
        odepth = (int) Math.ceil(1.0 * (inputdepth - dilationKernelDepth + 2 * padt) / dt) + 1;
      }
    } else {
      oheight = (int) Math.floor(1.0 * (inputHeight - dilationKernelHeight + 2 * padH) / dH) + 1;
      owidth = (int) Math.floor(1.0 * (inputWidth - dilationKernelWidth + 2 * padW) / dW) + 1;
      if (inputdepth > 0) {
        require(dt > 0 && kt > 0 && padt >= 0,
            "kernel size, stride size, padding size cannot be smaller than 0");
        odepth = (int) Math.floor(1.0 * (inputdepth - dilationKernelDepth + 2 * padt) / dt) + 1;
      }
    }

    if (padH != 0 || padW != 0 || padt != 0) {
      if ((oheight - 1) * dH >= inputHeight + padH) {
        oheight -= 1;
      }
      if ((owidth - 1) * dW >= inputWidth + padW) {
        owidth -= 1;
      }
      if (inputdepth > 0) {
        if ((odepth - 1) * dt >= inputdepth + padt) {
          odepth -= 1;
        }
        return new int[]{padt, padt, padH, padH, padW, padW, odepth, oheight, owidth};
      }
    } else if (inputdepth > 0) {
      return new int[]{padt, padt, padH, padH, padW, padW, odepth, oheight, owidth};
    }
    return new int[]{padH, padH, padW, padW, oheight, owidth};
  }

  /**
   * copyMaskAndScales.
   * @param from
   * @param to
   */
  public static void copyMaskAndScales(MemoryData[] from, MemoryData[] to) {
    // here we check the from and to, they should not be null ideally
    // but if the model is mixin with blas layer, it may be null
    if (from == null || to == null) {
      return;
    }

    boolean valid = (from.length == 1 || to.length == 1) || // the ConcatTable or JoinTable
        (from.length == to.length); // the same length of from and to.

    // if from has scales but to has no, copy them
    boolean needCopy = from != to
        && Arrays.stream(from).allMatch(memoryData -> memoryData.scales.length == 0)
        && Arrays.stream(to).allMatch(memoryData -> memoryData.scales.length == 0);

    if (valid && needCopy) {
      if (from.length == to.length) {
        for (int i = 0; i < from.length; i++) {
          MemoryData x1 = from[i];
          MemoryData x2 = to[i];
          if (x1.scales != null && x1.scales.length == 0) {
            x1.setScales(x2.scales);
            x1.setMask(x2.mask());
          }
        }
      } else if (to.length == 1) {
        float[][] temp = new float[from.length][from[0].scales.length];
        for (int i = 0; i < temp.length; i++) {
          temp[i] = from[i].scales;
        }

        float[] result = temp[0];
        for (int row = 1; row < temp.length; row++) {
          for (int column = 0; column < temp[0].length; column++) {
            if (temp[row][column] > result[column]) {
              result[column] = temp[row][column];
            }
          }
        }

        to[0].setScales(result);

        int maskTemp = from[0].mask();
        for (int i = 1; i < from.length; i++) {
          if (maskTemp != from[i].mask()) {
            require(false, "only support the same mask");
          }
        }

        to[0].setMask(maskTemp);
      } else if (to.length > 1) {
        Arrays.stream(to).forEach(memoryData -> memoryData.setScales(from[0].scales));
        Arrays.stream(to).forEach(memoryData -> memoryData.setMask(from[0].mask()));
      }
    }
  }

  /**
   * copyMaskAndScales.
   * @param from
   * @param to
   */
  public static void copyMaskAndScales(MemoryData from, MemoryData to) {
    // here we check the from and to, they should not be null ideally
    // but if the model is mixin with blas layer, it may be null
    if (from != null && to != null && to.scales.length == 0) {
      to.setScales(from.scales.clone());
      to.setMask(from.mask());
    }
  }
}
