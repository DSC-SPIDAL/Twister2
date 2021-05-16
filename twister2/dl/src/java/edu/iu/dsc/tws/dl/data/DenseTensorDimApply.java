//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless Util.Util.required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.
package edu.iu.dsc.tws.dl.data;

import edu.iu.dsc.tws.dl.data.function.TensorDimFunc2;
import edu.iu.dsc.tws.dl.data.function.TensorDimFunc3;
import edu.iu.dsc.tws.dl.data.tensor.DenseTensor;
import edu.iu.dsc.tws.dl.utils.Util;

@SuppressWarnings({"LocalVariableName", "ParameterName"})
/**
 * Utility class.
 */
public final class DenseTensorDimApply {

  private DenseTensorDimApply() {
  }

  /**
   * dimApply2.
   *
   * @param tensor1
   * @param tensor2
   * @param _dim
   * @param func
   */
  public static void dimApply2(DenseTensor tensor1,
                               DenseTensor tensor2, int _dim, TensorDimFunc2 func) {

    Util.require(_dim >= 0 && _dim < tensor1.nDimension(), "invalid dimension");
    Util.require(tensor1.nDimension() == tensor2.nDimension(), "inconsistent tensor sizes");
    int i = 0;
    while (i < tensor1.nDimension()) {
      if (i != _dim) {
        Util.require(tensor1.size(i + 1) == tensor2.size(i + 1), "inconsistent tensor sizes");
      }
      i += 1;
    }

    int[] counter = new int[tensor1.nDimension()];
    double[] _data1 = tensor1.storage().toDoubleArray();
    int _offset1 = tensor1.storageOffset() - 1;
    int stride1 = tensor1.stride(_dim + 1);
    int size1 = tensor1.size(_dim + 1);

    double[] _data2 = tensor2.storage().toDoubleArray();
    int _offset2 = tensor2.storageOffset() - 1;
    int stride2 = tensor2.stride(_dim + 1);
    int size2 = tensor2.size(_dim + 1);

    boolean hasFinished = false;
    while (!hasFinished) {
      func.apply(_data1, _offset1, stride1, size1,
          _data2, _offset2, stride2, size2);
      if (tensor1.nDimension() == 1) {
        hasFinished = true;
      } else {
        int j = 0;
        boolean breakCheck = false;
        while (j < tensor1.nDimension() && !breakCheck) {
          if (j == _dim) {
            if (j == tensor1.nDimension() - 1) {
              hasFinished = true;
              breakCheck = true;
            }
          } else {
            counter[j] += 1;
            _offset1 += tensor1.stride(j + 1);
            _offset2 += tensor2.stride(j + 1);

            if (counter[j] == tensor1.size(j + 1)) {
              if (j == tensor1.nDimension() - 1) {
                breakCheck = true;
                hasFinished = true;
              } else {
                _offset1 -= counter[j] * tensor1.stride(j + 1);
                _offset2 -= counter[j] * tensor2.stride(j + 1);
                counter[j] = 0;
              }
            } else {
              breakCheck = true;
            } // if (counter(i) == tensor1.size(i))
          } // if (i == _dim) else
          j += 1;
        } // while
      } // if(tensor1.nDimension() == 1)
    } // while(!hasFinished)
  }

  /**
   * dimApply3.
   *
   * @param tensor1
   * @param tensor2
   * @param tensor3
   * @param dim
   * @param func
   */
  public static void dimApply3(DenseTensor tensor1, DenseTensor tensor2, DenseTensor tensor3,
                               int dim, TensorDimFunc3 func) {
    Util.require(dim > 0 && dim <= tensor1.nDimension(), "invalid dimension");
    Util.require(tensor1.nDimension() == tensor2.nDimension(), "inconsistent tensor sizes");
    Util.require(tensor2.nDimension() == tensor3.nDimension(), "inconsistent tensor sizes");
    int d = 1;
    while (d <= tensor1.nDimension()) {
      if (d != dim) {
        Util.require(tensor1.size(d) == tensor2.size(d), "inconsistent tensor sizes");
        Util.require(tensor2.size(d) == tensor3.size(d), "inconsistent tensor sizes");
      }
      d += 1;
    }

    if (tensor1.isFloat()) {
      int[] counter = new int[tensor1.nDimension()];
      float[] _data1 = tensor1.storage().toFloatArray();
      int _offset1 = tensor1.storageOffset() - 1;
      int stride1 = tensor1.stride(dim);
      int size1 = tensor1.size(dim);

      float[] _data2 = tensor2.storage().toFloatArray();
      int _offset2 = tensor2.storageOffset() - 1;
      int stride2 = tensor2.stride(dim);
      int size2 = tensor2.size(dim);

      float[] _data3 = tensor3.storage().toFloatArray();
      int _offset3 = tensor3.storageOffset() - 1;
      int stride3 = tensor3.stride(dim);
      int size3 = tensor3.size(dim);

      boolean isFinished = false;

      while (!isFinished) {
        func.apply(_data1, _offset1, stride1, size1,
            _data2, _offset2, stride2, size2,
            _data3, _offset3, stride3, size3);

        if (tensor1.nDimension() == 1) {
          isFinished = true;
        } else {
          int k = 1;
          boolean isBreak = false;
          while (k <= tensor1.nDimension() && !isBreak) {
            if (k == dim) {
              if (k == tensor1.nDimension()) {
                isFinished = true;
                isBreak = true;
              }
            } else {
              counter[k - 1] += 1;
              _offset1 += tensor1.stride(k);
              _offset2 += tensor2.stride(k);
              _offset3 += tensor3.stride(k);

              if (counter[k - 1] == tensor1.size(k)) {
                if (k == tensor1.nDimension()) {
                  isFinished = true;
                  isBreak = true;
                } else {
                  _offset1 -= counter[k - 1] * tensor1.stride(k);
                  _offset2 -= counter[k - 1] * tensor2.stride(k);
                  _offset3 -= counter[k - 1] * tensor3.stride(k);
                  counter[k - 1] = 0;
                }
              } else {
                isBreak = true;
              }
            }
            k += 1;
          }
        }
      }
    } else {

      int[] counter = new int[tensor1.nDimension()];
      double[] _data1 = tensor1.storage().toDoubleArray();
      int _offset1 = tensor1.storageOffset() - 1;
      int stride1 = tensor1.stride(dim);
      int size1 = tensor1.size(dim);

      double[] _data2 = tensor2.storage().toDoubleArray();
      int _offset2 = tensor2.storageOffset() - 1;
      int stride2 = tensor2.stride(dim);
      int size2 = tensor2.size(dim);

      double[] _data3 = tensor3.storage().toDoubleArray();
      int _offset3 = tensor3.storageOffset() - 1;
      int stride3 = tensor3.stride(dim);
      int size3 = tensor3.size(dim);

      boolean isFinished = false;

      while (!isFinished) {
        func.apply(_data1, _offset1, stride1, size1,
            _data2, _offset2, stride2, size2,
            _data3, _offset3, stride3, size3);

        if (tensor1.nDimension() == 1) {
          isFinished = true;
        } else {
          int k = 1;
          boolean isBreak = false;
          while (k <= tensor1.nDimension() && !isBreak) {
            if (k == dim) {
              if (k == tensor1.nDimension()) {
                isFinished = true;
                isBreak = true;
              }
            } else {
              counter[k - 1] += 1;
              _offset1 += tensor1.stride(k);
              _offset2 += tensor2.stride(k);
              _offset3 += tensor3.stride(k);

              if (counter[k - 1] == tensor1.size(k)) {
                if (k == tensor1.nDimension()) {
                  isFinished = true;
                  isBreak = true;
                } else {
                  _offset1 -= counter[k - 1] * tensor1.stride(k);
                  _offset2 -= counter[k - 1] * tensor2.stride(k);
                  _offset3 -= counter[k - 1] * tensor3.stride(k);
                  counter[k - 1] = 0;
                }
              } else {
                isBreak = true;
              }
            }
            k += 1;
          }
        }
      }
    }


  }

}
