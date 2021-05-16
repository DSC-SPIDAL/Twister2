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
package edu.iu.dsc.tws.dl.utils.schema;

import java.lang.reflect.Array;

import edu.iu.dsc.tws.api.comms.messaging.types.MessageType;
import edu.iu.dsc.tws.api.comms.packing.DataPacker;
import edu.iu.dsc.tws.dl.utils.pair.DoubleDoubleArrayPair;

public class DoubleDoubleArrayType implements MessageType<DoubleDoubleArrayPair, double[]> {
  @Override
  public boolean isPrimitive() {
    return false;
  }

  @Override
  public int getUnitSizeInBytes() {
    return 0;
  }

  @Override
  public int getDataSizeInBytes(DoubleDoubleArrayPair data) {
    int size = Double.BYTES;
    size += Array.getLength(data.getValue1()) * Double.BYTES;
    return size;
  }

  @Override
  public Class getClazz() {
    return DoubleDoubleArrayPair.class;
  }

  @Override
  public DataPacker getDataPacker() {
    return DoubleDoubleArrayPacker.getInstance();
  }

  @Override
  public boolean isArray() {
    return false;
  }
}
