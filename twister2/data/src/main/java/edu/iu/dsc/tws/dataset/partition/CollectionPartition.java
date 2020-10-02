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

package edu.iu.dsc.tws.dataset.partition;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

import edu.iu.dsc.tws.api.dataset.DataPartition;
import edu.iu.dsc.tws.api.dataset.DataPartitionConsumer;
import edu.iu.dsc.tws.dataset.consumer.IterativeConsumer;

public class CollectionPartition<T> implements DataPartition<T> {

  protected List<T> dataList = new ArrayList<>();

  private int id;

  /**
   * This constructor is deprecated
   *
   * @deprecated use default constructor instead
   */
  @Deprecated
  public CollectionPartition(int id) {
    this.id = id;
  }

  public CollectionPartition() {

  }

  @Override
  public void setId(int id) {
    this.id = id;
  }

  public void add(T val) {
    dataList.add(val);
  }

  public void addAll(Collection<T> vals) {
    dataList.addAll(vals);
  }

  @Override
  public DataPartitionConsumer<T> getConsumer() {
    return new IterativeConsumer<>(dataList.iterator());
  }

  @Override
  public void clear() {
    this.dataList.clear();
  }

  @Override
  public int getPartitionId() {
    return id;
  }
}
