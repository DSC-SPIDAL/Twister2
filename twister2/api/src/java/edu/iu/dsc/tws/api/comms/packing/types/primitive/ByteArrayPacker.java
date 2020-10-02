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

package edu.iu.dsc.tws.api.comms.packing.types.primitive;

import java.nio.ByteBuffer;

import edu.iu.dsc.tws.api.comms.messaging.types.MessageType;
import edu.iu.dsc.tws.api.comms.messaging.types.MessageTypes;
import edu.iu.dsc.tws.api.comms.packing.DataBuffer;
import edu.iu.dsc.tws.api.comms.packing.ObjectBuilder;
import edu.iu.dsc.tws.api.comms.packing.PackerStore;

public final class ByteArrayPacker implements PrimitiveArrayPacker<byte[]> {

  private static volatile ByteArrayPacker instance;

  private ByteArrayPacker() {
  }

  public static ByteArrayPacker getInstance() {
    if (instance == null) {
      instance = new ByteArrayPacker();
    }
    return instance;
  }

  @Override
  public MessageType<byte[], byte[]> getMessageType() {
    return MessageTypes.BYTE_ARRAY;
  }

  @Override
  public ByteBuffer addToBuffer(ByteBuffer byteBuffer, byte[] data, int index) {
    return byteBuffer.put(data[index]);
  }

  @Override
  public ByteBuffer addToBuffer(ByteBuffer byteBuffer, int offset, byte[] data, int index) {
    return byteBuffer.put(offset, data[index]);
  }

  @Override
  public void readFromBufferAndSet(ByteBuffer byteBuffer, int offset, byte[] array, int index) {
    array[index] = byteBuffer.get(offset);
  }

  @Override
  public void readFromBufferAndSet(ByteBuffer byteBuffer, byte[] array, int index) {
    array[index] = byteBuffer.get();
  }

  @Override
  public void writeDataToBuffer(byte[] data, PackerStore packerStore, int alreadyCopied,
                                 int leftToCopy, int spaceLeft, ByteBuffer targetBuffer) {
    int unitSize = this.getMessageType().getUnitSizeInBytes();
    int elementsCopied = alreadyCopied / unitSize;
    int elementsLeft = leftToCopy / unitSize;

    int spacePermitsFor = spaceLeft / unitSize;
    int willCopy = Math.min(spacePermitsFor, elementsLeft);

    targetBuffer.put(data, elementsCopied, willCopy);
  }

  @Override
  public int readDataFromBuffer(ObjectBuilder objectBuilder, int currentBufferLocation,
                                DataBuffer dataBuffer) {
    int totalDataLength = objectBuilder.getTotalSize();
    int startIndex = objectBuilder.getCompletedSize();
    byte[] val = (byte[]) objectBuilder.getPartialDataHolder();

    ByteBuffer byteBuffer = dataBuffer.getByteBuffer();
    int remainingInBuffer = dataBuffer.getSize() - currentBufferLocation;
    int leftToRead = totalDataLength - startIndex;

    int elementsToRead = Math.min(leftToRead, remainingInBuffer);

    byteBuffer.position(currentBufferLocation); //setting position for bulk read
    byteBuffer.get(val, startIndex, elementsToRead);

    if (totalDataLength == elementsToRead + startIndex) {
      objectBuilder.setFinalObject(val);
    }
    return elementsToRead;
  }

  @Override
  public byte[] wrapperForLength(int length) {
    return new byte[length];
  }

  @Override
  public byte[] packToByteArray(byte[] data) {
    return data;
  }

  public byte[] unpackFromByteArray(byte[] array) {
    return array;
  }
}
