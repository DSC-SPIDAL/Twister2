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
package edu.iu.dsc.tws.dl.data;

import java.io.Serializable;
import java.util.List;

public interface MiniBatch extends Serializable {
  /**
   * Get the number of samples in this MiniBatch
   *
   * @return size How many samples in this MiniBatch
   */
  int size();

  /**
   * Slice this MiniBatch to a smaller MiniBatch with offset and length
   *
   * @param offset offset, counted from 1
   * @param length length
   * @return A smaller MiniBatch
   */
  MiniBatch slice(int offset, int length);

  /**
   * Get input in this MiniBatch.
   *
   * @return input Activity
   */
  Activity getInput();

  /**
   * Get target in this MiniBatch
   *
   * @return target Activity
   */
  Activity getTarget();

  /**
   * Replace the original content of the miniBatch with a set of Sample.
   *
   * @param samples a set of Sample
   * @return self
   */
  MiniBatch set(List<Sample> samples);
}
