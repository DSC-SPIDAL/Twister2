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

package edu.iu.dsc.tws.api.comms.messaging;

import java.util.ArrayList;
import java.util.List;

import edu.iu.dsc.tws.api.comms.messaging.types.MessageType;
import edu.iu.dsc.tws.api.comms.messaging.types.MessageTypes;
import edu.iu.dsc.tws.api.comms.packing.DataBuffer;

public class ChannelMessage {
  /**
   * List of buffers filled with the message
   */
  private final List<DataBuffer> buffers = new ArrayList<>();

  /**
   * List of byte arrays which are used to copy data from {@link ChannelMessage#buffers}
   * When the system runs out of receive buffers
   */
  private final List<DataBuffer> overflowBuffers = new ArrayList<>();

  /**
   * Keeps the number of references to this message
   * The resources associated with the message is released when refcount becomes 0
   */
  private int refCount;

  /**
   * Type of the message, weather request or send
   */
  private MessageDirection messageDirection;

  /**
   * We call this to release the buffers
   */
  private ChannelMessageReleaseCallback releaseListener;

  /**
   * Keep track of the originating id, this is required to release the buffers allocated.
   */
  private int originatingId;

  /**
   * The message header
   */
  protected MessageHeader header;

  /**
   * Keeps track of whether header of the object contained in the buffer was sent or not.
   * This is only used when a message is broken down into several buffers and each buffer is sent
   * separately
   */
  private boolean headerSent;

  /**
   * Keep whether the current message is a partial object. This depends on whether all the data
   * is copied into the buffers or not. This is used when large messages are broken down into
   * several smaller messages
   */
  private boolean isPartial;

  /**
   * Keep whether we have all the buffers added
   */
  protected boolean complete = false;

  /**
   * Message type
   */
  private MessageType dataType;

  /**
   * If a keyed message, the key being used
   */
  private MessageType keyType = MessageTypes.INTEGER;

  /**
   * Number of bytes in the header
   */
  private int headerSize;

  /**
   * Keep track of accepted external sends
   */
  private int acceptedExternalSends = 0;

  /**
   * Keep track weather out count updated
   */
  private boolean outCountUpdated = false;


  public ChannelMessage() {
  }

  public ChannelMessage(int originatingId, MessageType messageType,
                        MessageDirection messageDirection,
                        ChannelMessageReleaseCallback releaseListener) {
    this.refCount = 0;
    this.messageDirection = messageDirection;
    this.releaseListener = releaseListener;
    this.originatingId = originatingId;
    this.complete = false;
    this.dataType = messageType;
  }

  public List<DataBuffer> getBuffers() {
    if (overflowBuffers.size() > 0) {
      List<DataBuffer> total = new ArrayList<>();
      total.addAll(overflowBuffers);
      total.addAll(buffers);
      return total;
    } else {
      return buffers;
    }
  }

  /**
   * returns the direct buffers that were allocated.
   */
  public List<DataBuffer> getNormalBuffers() {
    return buffers;
  }

  public void incrementRefCount() {
    refCount++;
  }

  public void incrementRefCount(int count) {
    refCount += count;
  }

  public MessageDirection getMessageDirection() {
    return messageDirection;
  }

  public boolean doneProcessing() {
    return refCount == 0;
  }

  /**
   * Release the allocated resources to this buffer.
   */
  public void release() {
    refCount--;
    if (refCount == 0) {
      releaseListener.release(this);
    }
  }

  public void addBuffer(DataBuffer buffer) {
    buffers.add(buffer);
  }

  public void addBuffers(List<DataBuffer> bufferList) {
    buffers.addAll(bufferList);
  }

  public void addOverFlowBuffers(List<DataBuffer> bufferList) {
    overflowBuffers.addAll(bufferList);
  }

  protected void removeAllBuffers() {
    overflowBuffers.clear();
    buffers.clear();
  }

  public void addToOverFlowBuffer(DataBuffer data) {
    overflowBuffers.add(data);
  }

  public List<DataBuffer> getOverflowBuffers() {
    return overflowBuffers;
  }

  public int getOriginatingId() {
    return originatingId;
  }

  public MessageHeader getHeader() {
    return header;
  }

  public void setHeader(MessageHeader header) {
    this.header = header;
  }

  public void setOriginatingId(int originatingId) {
    this.originatingId = originatingId;
  }

  public void setMessageDirection(MessageDirection messageDirection) {
    this.messageDirection = messageDirection;
  }

  public ChannelMessageReleaseCallback getReleaseListener() {
    return releaseListener;
  }

  public void setReleaseListener(ChannelMessageReleaseCallback releaseListener) {
    this.releaseListener = releaseListener;
  }

  public void setDataType(MessageType dataType) {
    this.dataType = dataType;
  }

  public boolean isComplete() {
    return complete;
  }

  public void setComplete(boolean complete) {
    this.complete = complete;
  }

  public MessageType getDataType() {
    return dataType;
  }

  public void setHeaderSize(int headerSize) {
    this.headerSize = headerSize;
  }

  public int getHeaderSize() {
    return headerSize;
  }

  public void setKeyType(MessageType keyType) {
    this.keyType = keyType;
  }

  public MessageType getKeyType() {
    return keyType;
  }

  public boolean isHeaderSent() {
    return headerSent;
  }

  public void setHeaderSent(boolean headerSent) {
    this.headerSent = headerSent;
  }

  public boolean isPartial() {
    return isPartial;
  }

  public void setPartial(boolean partial) {
    isPartial = partial;
  }

  public int getAcceptedExternalSends() {
    return acceptedExternalSends;
  }

  public int incrementAcceptedExternalSends() {
    return ++acceptedExternalSends;
  }

  public void setOutCountUpdated(boolean outCountUpdated) {
    this.outCountUpdated = outCountUpdated;
  }

  public boolean isOutCountUpdated() {
    return outCountUpdated;
  }
}
