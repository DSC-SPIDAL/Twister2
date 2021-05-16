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
import java.nio.ByteBuffer;

import edu.iu.dsc.tws.api.comms.packing.DataBuffer;
import edu.iu.dsc.tws.api.comms.packing.DataPacker;
import edu.iu.dsc.tws.api.comms.packing.ObjectBuilder;
import edu.iu.dsc.tws.api.comms.packing.PackerStore;
import edu.iu.dsc.tws.api.comms.packing.types.primitive.DoubleArrayPacker;
import edu.iu.dsc.tws.dl.utils.pair.DoubleDoubleArrayPair;

public final class DoubleDoubleArrayPacker implements DataPacker<DoubleDoubleArrayPair,
    DoubleDoubleArrayPair> {

  private static volatile DoubleDoubleArrayPacker instance;
  private DoubleArrayPacker arrayPacker;

  private DoubleDoubleArrayPacker() {
    this.arrayPacker = DoubleArrayPacker.getInstance();
  }

  public static DoubleDoubleArrayPacker getInstance() {
    if (instance == null) {
      instance = new DoubleDoubleArrayPacker();
    }
    return instance;
  }

  @Override
  public int determineLength(DoubleDoubleArrayPair data, PackerStore store) {
    int size = Double.BYTES;
    size += Array.getLength(data.getValue1()) * Double.BYTES;
    return size;
  }

  @Override
  public void writeDataToBuffer(DoubleDoubleArrayPair data, PackerStore packerStore,
                                int alreadyCopied, int leftToCopy, int spaceLeft,
                                ByteBuffer targetBuffer) {
    if (alreadyCopied < Double.BYTES) {
      //Need to write the loss value
      targetBuffer.putDouble(data.getValue0());
      arrayPacker.writeDataToBuffer(data.getValue1(), packerStore, 0,
          leftToCopy - Double.BYTES, spaceLeft - Double.BYTES, targetBuffer);
    } else {
      arrayPacker.writeDataToBuffer(data.getValue1(), packerStore,
          alreadyCopied - Double.BYTES, leftToCopy, spaceLeft, targetBuffer);
    }
  }

  public boolean bulkReadFromBuffer(ByteBuffer buffer, double[] dest, int offset, int length) {
    buffer.asDoubleBuffer().get(dest, offset, length);
    return true;
  }

  public void readFromBufferAndSet(ByteBuffer byteBuffer, int offset, double[] array, int index) {
    array[index] = byteBuffer.getDouble(offset);
  }

  @Override
  public int readDataFromBuffer(ObjectBuilder<DoubleDoubleArrayPair,
      DoubleDoubleArrayPair> objectBuilder, int currentBufferLocation, DataBuffer dataBuffer) {
    int totalDataLength = objectBuilder.getTotalSize();
    int startIndex = objectBuilder.getCompletedSize();
    int arrayStartIndex = 0;
    DoubleDoubleArrayPair val = objectBuilder.getPartialDataHolder();
    ByteBuffer byteBuffer = dataBuffer.getByteBuffer();
    int bufferPosition = currentBufferLocation;
    int bytesRead = 0;
    int lossSize = Double.BYTES;
    if (startIndex <= lossSize) {
      //Need to read in the loss value
      val.setValue0(byteBuffer.getDouble(bufferPosition));
      bytesRead += lossSize;
      bufferPosition += lossSize;
      arrayStartIndex = 0;
    } else {
      arrayStartIndex = (startIndex - lossSize) / Double.BYTES;
    }
    int totalNoOfElements = totalDataLength - lossSize / Double.BYTES;
    int size = dataBuffer.getSize();
    int elementsLeftInBuffer = (size - bufferPosition) / Double.BYTES;
    int noOfElementsToRead = Math.min(totalNoOfElements - arrayStartIndex, elementsLeftInBuffer);

    // first try to bulk read if child implementation supports it
    byteBuffer.position(currentBufferLocation);
    if (!bulkReadFromBuffer(byteBuffer, val.getValue1(), arrayStartIndex, noOfElementsToRead)) {
      for (int i = arrayStartIndex; i < totalNoOfElements; i++) {
        int remaining = size - bufferPosition;
        if (remaining >= Double.BYTES) {
          this.readFromBufferAndSet(byteBuffer, bufferPosition, val.getValue1(), i);
          bytesRead += Double.BYTES;
          bufferPosition += Double.BYTES;
        } else {
          break;
        }
      }
    } else {
      bytesRead += Double.BYTES * noOfElementsToRead;
    }

    int currentTotal = bytesRead + arrayStartIndex * Double.BYTES;
    if (arrayStartIndex > 0) {
      currentTotal += lossSize;
    }

    if (totalDataLength == currentTotal) {
      objectBuilder.setFinalObject(val);
    }
    return bytesRead;
  }

  @Override
  public byte[] packToByteArray(DoubleDoubleArrayPair data) {
    int size = Double.BYTES;
    size += Array.getLength(data.getValue1()) * Double.BYTES;
    byte[] byteArray = new byte[size];
    ByteBuffer wrapper = ByteBuffer.wrap(byteArray);
    this.packToByteBuffer(wrapper, data);
    return byteArray;
  }

  @Override
  public ByteBuffer packToByteBuffer(ByteBuffer byteBuffer, DoubleDoubleArrayPair data) {
    byteBuffer.putDouble(data.getValue0());
    for (int i = 0; i < Array.getLength(data); i++) {
      byteBuffer.putDouble(data.getValue1()[i]);
    }
    return byteBuffer;
  }

  @Override
  public ByteBuffer packToByteBuffer(ByteBuffer byteBuffer, int offset,
                                     DoubleDoubleArrayPair data) {
    byteBuffer.putDouble(data.getValue0());
    for (int i = 0; i < Array.getLength(data.getValue1()); i++) {
      byteBuffer.putDouble(offset + i * Double.BYTES, data.getValue1()[i]);
    }
    return byteBuffer;
  }

  @Override
  public DoubleDoubleArrayPair wrapperForByteLength(int byteLength) {
    int arraySize = (byteLength - Double.BYTES) / Double.BYTES;
    return new DoubleDoubleArrayPair(0.0, new double[arraySize]);
  }

  @Override
  public boolean isHeaderRequired() {
    return true;
  }

  @Override
  public DoubleDoubleArrayPair unpackFromBuffer(ByteBuffer byteBuffer, int bufferOffset,
                                                int byteLength) {
    int initialBufferPosition = byteBuffer.position();
    DoubleDoubleArrayPair result = this.wrapperForByteLength(byteLength);
    result.setValue0(byteBuffer.getDouble(initialBufferPosition + bufferOffset));
    int noOfElements = byteLength - Double.BYTES / Double.BYTES;
    int unitSize = Double.BYTES;
    initialBufferPosition = byteBuffer.position();
    for (int i = 0; i < noOfElements; i++) {
      result.getValue1()[i] = byteBuffer
          .getDouble(bufferOffset + initialBufferPosition + i * unitSize);
    }
    return result;
  }

  @Override
  public DoubleDoubleArrayPair unpackFromBuffer(ByteBuffer byteBuffer, int byteLength) {
    DoubleDoubleArrayPair result = this.wrapperForByteLength(byteLength);
    result.setValue0(byteBuffer.getDouble());
    int noOfElements = byteLength - Double.BYTES / Double.BYTES;
    for (int i = 0; i < noOfElements; i++) {
      result.getValue1()[i] = byteBuffer.getDouble();
    }
    return result;
  }
}
