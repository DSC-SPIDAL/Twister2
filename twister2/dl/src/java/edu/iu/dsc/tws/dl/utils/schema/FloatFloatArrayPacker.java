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
import edu.iu.dsc.tws.api.comms.packing.types.primitive.FloatArrayPacker;
import edu.iu.dsc.tws.dl.utils.pair.FloatFloatArrayPair;

public final class FloatFloatArrayPacker implements DataPacker<FloatFloatArrayPair,
    FloatFloatArrayPair> {

  private static volatile FloatFloatArrayPacker instance;
  private FloatArrayPacker arrayPacker;

  private FloatFloatArrayPacker() {
    this.arrayPacker = FloatArrayPacker.getInstance();
  }

  public static FloatFloatArrayPacker getInstance() {
    if (instance == null) {
      instance = new FloatFloatArrayPacker();
    }
    return instance;
  }

  @Override
  public int determineLength(FloatFloatArrayPair data, PackerStore store) {
    int size = Float.BYTES;
    size += Array.getLength(data.getValue1()) * Float.BYTES;
    return size;
  }

  @Override
  public void writeDataToBuffer(FloatFloatArrayPair data, PackerStore packerStore,
                                int alreadyCopied, int leftToCopy, int spaceLeft,
                                ByteBuffer targetBuffer) {
    if (alreadyCopied < Float.BYTES) {
      //Need to write the loss value
      targetBuffer.putFloat(data.getValue0());
      arrayPacker.writeDataToBuffer(data.getValue1(), packerStore, 0,
          leftToCopy - Float.BYTES, spaceLeft - Float.BYTES, targetBuffer);
    } else {
      arrayPacker.writeDataToBuffer(data.getValue1(), packerStore,
          alreadyCopied - Float.BYTES, leftToCopy, spaceLeft, targetBuffer);
    }
  }

  public boolean bulkReadFromBuffer(ByteBuffer buffer, float[] dest, int offset, int length) {
    buffer.asFloatBuffer().get(dest, offset, length);
    return true;
  }

  public void readFromBufferAndSet(ByteBuffer byteBuffer, int offset, float[] array, int index) {
    array[index] = byteBuffer.getFloat(offset);
  }

  @Override
  public int readDataFromBuffer(ObjectBuilder<FloatFloatArrayPair,
      FloatFloatArrayPair> objectBuilder, int currentBufferLocation, DataBuffer dataBuffer) {
    int totalDataLength = objectBuilder.getTotalSize();
    int startIndex = objectBuilder.getCompletedSize();
    int arrayStartIndex = 0;
    FloatFloatArrayPair val = objectBuilder.getPartialDataHolder();
    ByteBuffer byteBuffer = dataBuffer.getByteBuffer();
    int bufferPosition = currentBufferLocation;
    int bytesRead = 0;
    int lossSize = Float.BYTES;
    if (startIndex <= lossSize) {
      //Need to read in the loss value
      val.setValue0(byteBuffer.getFloat(bufferPosition));
      bytesRead += lossSize;
      bufferPosition += lossSize;
      arrayStartIndex = 0;
    } else {
      arrayStartIndex = (startIndex - lossSize) / Float.BYTES;
    }
    int totalNoOfElements = totalDataLength - lossSize / Float.BYTES;
    int size = dataBuffer.getSize();
    int elementsLeftInBuffer = (size - bufferPosition) / Float.BYTES;
    int noOfElementsToRead = Math.min(totalNoOfElements - arrayStartIndex, elementsLeftInBuffer);

    // first try to bulk read if child implementation supports it
    byteBuffer.position(currentBufferLocation);
    if (!bulkReadFromBuffer(byteBuffer, val.getValue1(), arrayStartIndex, noOfElementsToRead)) {
      for (int i = arrayStartIndex; i < totalNoOfElements; i++) {
        int remaining = size - bufferPosition;
        if (remaining >= Float.BYTES) {
          this.readFromBufferAndSet(byteBuffer, bufferPosition, val.getValue1(), i);
          bytesRead += Float.BYTES;
          bufferPosition += Float.BYTES;
        } else {
          break;
        }
      }
    } else {
      bytesRead += Float.BYTES * noOfElementsToRead;
    }

    int currentTotal = bytesRead + arrayStartIndex * Float.BYTES;
    if (arrayStartIndex > 0) {
      currentTotal += lossSize;
    }

    if (totalDataLength == currentTotal) {
      objectBuilder.setFinalObject(val);
    }
    return bytesRead;
  }

  @Override
  public byte[] packToByteArray(FloatFloatArrayPair data) {
    int size = Float.BYTES;
    size += Array.getLength(data.getValue1()) * Float.BYTES;
    byte[] byteArray = new byte[size];
    ByteBuffer wrapper = ByteBuffer.wrap(byteArray);
    this.packToByteBuffer(wrapper, data);
    return byteArray;
  }

  @Override
  public ByteBuffer packToByteBuffer(ByteBuffer byteBuffer, FloatFloatArrayPair data) {
    byteBuffer.putFloat(data.getValue0());
    for (int i = 0; i < Array.getLength(data); i++) {
      byteBuffer.putFloat(data.getValue1()[i]);
    }
    return byteBuffer;
  }

  @Override
  public ByteBuffer packToByteBuffer(ByteBuffer byteBuffer, int offset,
                                     FloatFloatArrayPair data) {
    byteBuffer.putFloat(data.getValue0());
    for (int i = 0; i < Array.getLength(data.getValue1()); i++) {
      byteBuffer.putFloat(offset + i * Float.BYTES, data.getValue1()[i]);
    }
    return byteBuffer;
  }

  @Override
  public FloatFloatArrayPair wrapperForByteLength(int byteLength) {
    int arraySize = (byteLength - Float.BYTES) / Float.BYTES;
    return new FloatFloatArrayPair(0.0f, new float[arraySize]);
  }

  @Override
  public boolean isHeaderRequired() {
    return true;
  }

  @Override
  public FloatFloatArrayPair unpackFromBuffer(ByteBuffer byteBuffer, int bufferOffset,
                                                int byteLength) {
    int initialBufferPosition = byteBuffer.position();
    FloatFloatArrayPair result = this.wrapperForByteLength(byteLength);
    result.setValue0(byteBuffer.getFloat(initialBufferPosition + bufferOffset));
    int noOfElements = byteLength - Float.BYTES / Float.BYTES;
    int unitSize = Float.BYTES;
    initialBufferPosition = byteBuffer.position();
    for (int i = 0; i < noOfElements; i++) {
      result.getValue1()[i] = byteBuffer
          .getFloat(bufferOffset + initialBufferPosition + i * unitSize);
    }
    return result;
  }

  @Override
  public FloatFloatArrayPair unpackFromBuffer(ByteBuffer byteBuffer, int byteLength) {
    FloatFloatArrayPair result = this.wrapperForByteLength(byteLength);
    result.setValue0(byteBuffer.getFloat());
    int noOfElements = byteLength - Float.BYTES / Float.BYTES;
    for (int i = 0; i < noOfElements; i++) {
      result.getValue1()[i] = byteBuffer.getFloat();
    }
    return result;
  }
}
