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
import java.util.HashMap;
import java.util.Map;
import java.util.Set;
import java.util.function.BiConsumer;

/**
 * This is a half baked version of the Table in BigDL framework
 * More work needs to be done to make it compatible
 */
public class Table implements Activity, Serializable {
  private Map state = new HashMap();
  // index of last element in the contiguous numeric number indexed elements start from 1
  private int topIndex = 0;

  public Table() {
  }

  public Table(Object[] data) {
    while (topIndex < data.length) {
      state.put(topIndex + 1, data[topIndex]);
      topIndex += 1;
    }
  }

  @Override
  public Tensor toTensor() {
    return null;
  }

  @Override
  public Table toTable() {
    return null;
  }

  @Override
  public boolean isTensor() {
    return false;
  }

  @Override
  public boolean isTable() {
    return true;
  }

  /**
   * Empty the Table
   */
  public Table clear() {
    state.clear();
    topIndex = 0;
    return this;
  }

  public void forEach(BiConsumer func) {
    state.forEach(func);
  }

  public void put(Object key, Object value) {
    state.put(key, value);
  }

  public Set keySet() {
    return this.state.keySet();
  }

  public <T> T get(int key) {
    return (T) state.get(key);
  }

  public <T> T get(Object key) {
    return (T) state.get(key);
  }

  public <T> T getOrElse(int key, T defaultVal) {
    return (T) state.getOrDefault(key, defaultVal);
  }

  public boolean contains(int key) {
    return state.containsKey(key);
  }

  public boolean contains(Object key) {
    return state.containsKey(key);
  }

  public <T> T apply(int key) {
    return (T) state.get(key);
  }

  public Table update(int key, Object value) {
    state.put(key, value);
    if (topIndex + 1 == key) {
      topIndex += 1;
      while (state.containsKey(topIndex + 1)) {
        topIndex += 1;
      }
    }
    return this;
  }

//  @Override
//  public Table clone() throws CloneNotSupportedException {
//    Table clone = new Table();
//    for (Object key : state.keySet()) {
//      clone.update((int) key, state.get(key));
//    }
//    return clone;
//  }

  @Override
  public String toString() {
    return super.toString();
  }

  @Override
  public boolean equals(Object obj) {
    if (obj == null) {
      return false;
    }
    if (!(obj instanceof Table)) {
      return false;
    }
    Table other = (Table) obj;
    if (this == other) {
      return true;
    }
    if (this.state.keySet().size() != other.state.keySet().size()) {
      return false;
    }
    for (Object key : state.keySet()) {
      if (this.state.get(key).getClass().isArray() && other.state.get(key).getClass().isArray()) {
        //TODO add support to check array equality
        throw new IllegalStateException("Array type Not supported yet for equals check");
      } else if (this.state.get(key) != other.state.get(key)) {
        return false;
      }
    }
    return true;
  }

  @Override
  public int hashCode() {
    int seed = 37;
    int hash = 1;

    for (Object key : state.keySet()) {
      hash = hash * seed + key.hashCode();
      hash = hash * seed + this.state.get(key).hashCode();
    }
    return hash;
  }

  public <T> T remove(int index) {
    if (index < 0) {
      return null;
    }

    if (topIndex >= index) {
      int i = index;
      Object result = state.get(index);
      while (i < topIndex) {
        state.put(i, state.get(i + 1));
        i += 1;
      }
      state.remove(topIndex);
      topIndex -= 1;
      return (T) result;
    } else if (state.containsKey(index)) {
      return (T) state.remove(index);
    } else {
      return null;
    }
  }

  public <T> T remove() {
    if (topIndex != 0) {
      return remove(topIndex);
    } else {
      return null;
    }
  }

  public <T> Table insert(T obj) {
    return update(topIndex + 1, obj);
  }

  public <T> Table insert(int index, T obj) {
    if (index < 0) {
      return null;
    }

    if (topIndex >= index) {
      int i = topIndex + 1;
      topIndex += 1;
      while (i > index) {
        state.put(i, state.get(i - 1));
        i -= 1;
      }
      update(index, obj);
    } else {
      update(index, obj);
    }

    return this;
  }

  public Table delete(Object key) {
    if (state.containsKey(key)) {
      state.remove(key);
    }
    return this;
  }

  public int length() {
    return state.size();
  }

  public Table save(String path, boolean overWrite) {
    //TODO: needs impl
    throw new IllegalStateException("Not implemented yet");
    //File.save(this, path, overWrite);
    //return this;
  }

  /**
   * Recursively flatten the table to a single table containing no nested table inside
   *
   * @return the flatten table
   */
  public Table flatten() {
    return flatten(1);
  }

  private Table flatten(int startIndex) {
    //TODO: needs impl
    throw new IllegalStateException("Not implemented yet");
  }

  /**
   * Recursively inverse flatten the flatten table to the same shape with target
   *
   * @param target the target shape to become
   * @return the inverse flatten the table with the same shape with target
   */
  public Table inverseFlatten(Table target) {
    return inverseFlatten(target, 1);
  }

  /**
   * Recursively inverse flatten the flatten table to the same shape with target
   *
   * @param target     the target shape to become
   * @param startIndex for each iteration the start index as an offset
   * @return the inverse flatten the table with the same shape with target
   */
  private Table inverseFlatten(Table target, int startIndex) {
    //TODO: needs impl
    throw new IllegalStateException("Not implemented yet");
  }

  public <T> T getOrDefault(Object key, T defaultValue) {
    return (T) state.getOrDefault(key, defaultValue);
  }

}
