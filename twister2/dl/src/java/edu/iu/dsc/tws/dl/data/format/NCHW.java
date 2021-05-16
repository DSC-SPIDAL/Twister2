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
package edu.iu.dsc.tws.dl.data.format;

/**
 * Specify the input/output data format when data is stored in the order of
 * [batch, channels, height, width]
 */
public class NCHW extends DataFormat {

  @Override
  public int[] getHWCDims(int inputDims) {
    if (inputDims == 3) {
      return new int[]{2, 3, 1};
    } else {
      return new int[]{3, 4, 2};
    }
  }
}
