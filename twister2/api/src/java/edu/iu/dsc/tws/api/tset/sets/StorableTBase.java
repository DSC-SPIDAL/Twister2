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

package edu.iu.dsc.tws.api.tset.sets;

import java.io.Serializable;
import java.util.List;

import edu.iu.dsc.tws.api.dataset.DataObject;
import edu.iu.dsc.tws.api.tset.TBase;

/**
 * All {@link edu.iu.dsc.tws.api.tset.sets.TSet}s that store data would need to implement this
 * interface. This defines the methods that other classes can use to access the stored data. Once
 * the data is stored, it would be exposed to as another "SourceTSet".
 */
public interface StorableTBase<T> extends TBase, Serializable {

  /**
   * retrieve data saved in the TSet as a {@link List}
   * <p>
   * NOTE: use this method only when you need to pull the data from the tset. Otherwise
   * this would unnecessarily loads data to the memory.
   *
   * @return dataObject
   */
  List<T> getData();


  default DataObject<T> getDataObject() {
    return null;
  }

  /**
   * retrieve data saved in the TSet
   *
   * @return dataObject
   */
  TBase getStoredSourceTSet();
}
