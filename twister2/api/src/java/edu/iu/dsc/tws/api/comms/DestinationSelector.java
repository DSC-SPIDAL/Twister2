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
package edu.iu.dsc.tws.api.comms;

import java.util.Set;

import edu.iu.dsc.tws.api.comms.messaging.types.MessageType;

/**
 * Destination selector interface needs to be implemented when creating destination selection
 * logic. For example for a keyed operation a destination selector will be used to calculate the
 * correct destination based on the key values.
 */
public interface DestinationSelector {
  /**
   * Prepare the destination selector
   *
   * @param comm the communicator
   * @param sources sources
   * @param destinations destination
   */
  default void prepare(Communicator comm, Set<Integer> sources, Set<Integer> destinations) {
  };

  /**
   * Prepare the destination selector, if this method is not overridden to support keyType
   * and dataType those values will be simply discarded.
   *
   * @param comm the communicator
   * @param sources sources
   * @param destinations destination
   * @param keyType type of key that this selector handles
   * @param dataType type of data that this selector handles
   */
  default void prepare(Communicator comm, Set<Integer> sources, Set<Integer> destinations,
                       MessageType keyType, MessageType dataType) {
    prepare(comm, sources, destinations);
  }

  /**
   * Get next destination using source and the data
   *
   * @param source source
   * @param data data
   * @return the next destination
   */
  int next(int source, Object data);

  /**
   * Get next destination using source, key and data
   *
   * @param source source
   * @param key key
   * @param data data
   * @return the next destination
   */
  default int next(int source, Object key, Object data) {
    return 0;
  }

  /**
   * Say that we have used the obtained destination
   *
   * @param source source
   * @param obtained obtained destination
   */
  default void commit(int source, int obtained) {
  }
}
