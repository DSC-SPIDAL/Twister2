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
package edu.iu.dsc.tws.column;

import org.apache.arrow.vector.types.pojo.Schema;

import edu.iu.dsc.tws.api.comms.messaging.types.MessageType;
import edu.iu.dsc.tws.api.comms.packing.DataPacker;

public class ArrowTableType implements MessageType {
  private Schema schema;

  public ArrowTableType(Schema schema) {
    this.schema = schema;
  }

  @Override
  public boolean isPrimitive() {
    return false;
  }

  @Override
  public int getUnitSizeInBytes() {
    return 0;
  }

  @Override
  public int getDataSizeInBytes(Object data) {
    return 0;
  }

  @Override
  public Class getClazz() {
    return ArrowTable.class;
  }

  @Override
  public DataPacker getDataPacker() {
    return new ArrowTablePacker(schema);
  }

  @Override
  public boolean isArray() {
    return false;
  }
}
