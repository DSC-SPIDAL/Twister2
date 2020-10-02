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
package edu.iu.dsc.tws.comms.dfw;

import java.util.LinkedList;
import java.util.List;
import java.util.Queue;
import java.util.logging.Logger;

import edu.iu.dsc.tws.api.comms.messaging.ChannelMessage;
import edu.iu.dsc.tws.api.comms.messaging.ChannelMessageReleaseCallback;
import edu.iu.dsc.tws.api.comms.messaging.MessageHeader;
import edu.iu.dsc.tws.api.comms.messaging.types.MessageType;
import edu.iu.dsc.tws.api.comms.messaging.types.MessageTypes;
import edu.iu.dsc.tws.api.comms.packing.DataBuffer;
import edu.iu.dsc.tws.api.comms.structs.Tuple;
import edu.iu.dsc.tws.comms.dfw.io.AggregatedObjects;
import edu.iu.dsc.tws.comms.dfw.io.DFWIOUtils;
import edu.iu.dsc.tws.comms.dfw.io.ObjectBuilderImpl;

public class InMessage {
  private static final Logger LOG = Logger.getLogger(InMessage.class.getName());

  public enum ReceivedState {
    INIT,
    BUILDING,
    BUILT,
    RECEIVE,
    DONE,
  }

  /**
   * The channels built
   */
  private Queue<ChannelMessage> builtMessages = new LinkedList<>();

  /**
   * The buffers added to this message
   */
  private Queue<DataBuffer> buffers = new LinkedList<>();

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
   * Keep whether we have all the buffers added
   */
  protected boolean complete;

  /**
   * Message type
   */
  private MessageType dataType;

  /**
   * If a keyed message, the key being used
   */
  private MessageType keyType = MessageTypes.INTEGER;

  /**
   * The deserialized data
   */
  private Object deserializedData;

  /**
   * Number of buffers added
   */
  private int addedBuffers = 0;

  // the amount of data we have seen for current object
  private int bufferPreviousReadForObject = 0;

  // keep track of the current object length
  private int bufferCurrentObjectLength = 0;

  // the objects we have in buffers so far, this doesn't mean we have un-packed them
  private int bufferSeenObjects = 0;

  /**
   * The length of the total object
   */
  private int unPkCurrentObjectLength = 0;

  /**
   * The length of the key unpacked
   */
  private int unPkCurrentKeyLength = -1;

  /**
   * The number of objects unpacked
   */
  private int unPkNumberObjects = 0;

  /**
   * Number of buffers we have unpacked
   */
  private int unPkBuffers = 0;

  /**
   * Weather we are reading the key
   */
  private boolean readingKey = true;

  /**
   * Weather this is a keyed message
   */
  private boolean keyed;


  /**
   * Received state
   */
  private ReceivedState receivedState;

  /**
   * The worker id
   */
  private int workerId;

  /**
   * This will be passed to packers to build objects for this message
   * todo check if this class can be simplified due to dataBuilder.
   */
  private ObjectBuilderImpl dataBuilder = new ObjectBuilderImpl();
  private ObjectBuilderImpl keyBuilder = new ObjectBuilderImpl();

  public InMessage(int originatingId, MessageType messageType,
                   ChannelMessageReleaseCallback releaseListener,
                   MessageHeader header) {
    this.releaseListener = releaseListener;
    this.originatingId = originatingId;
    this.complete = false;
    this.dataType = messageType;
    this.receivedState = ReceivedState.INIT;
    this.header = header;
    if (header.getNumberTuples() > 0) {
      deserializedData = new AggregatedObjects<>();
    }
  }

  public void setDataType(MessageType dataType) {
    this.dataType = dataType;
  }

  public MessageType getDataType() {
    return dataType;
  }

  public void setKeyType(MessageType keyType) {
    this.keyType = keyType;
  }

  public MessageType getKeyType() {
    keyed = true;
    return keyType;
  }

  public MessageHeader getHeader() {
    return header;
  }

  /**
   * Add a buffer and calculate weather we have seen all the buffers for an object
   *
   * @param buffer buffer
   * @return true if all the buffers for a message is received
   */
  public boolean addBufferAndCalculate(DataBuffer buffer) {
    addedBuffers++;

    byte lastBuffer;
    if (addedBuffers == 1) {
      lastBuffer = buffer.getByteBuffer().get(DFWIOUtils.HEADER_SIZE - 1);
    } else {
      lastBuffer = buffer.getByteBuffer().get(DFWIOUtils.SHORT_HEADER_SIZE - 1);
    }
    buffers.add(buffer);
    return lastBuffer == 1;
  }

  @SuppressWarnings("unchecked")
  public void addCurrentKeyedObject() {
    Object key = keyBuilder.getFinalObject();
    Object value = dataBuilder.getFinalObject();
    if (header.getNumberTuples() == -1) {
      deserializedData = new Tuple(key, value);
    } else {
      ((List<Object>) deserializedData).add(new Tuple(key, value));
    }
    unPkNumberObjects++;
    this.setUnPkCurrentObjectLength(-1);
    this.setUnPkCurrentKeyLength(-1);
  }


  public void addCurrentObject() {
    Object value = dataBuilder.getFinalObject();
    if (header.getNumberTuples() == -1) {
      deserializedData = value;
    } else {
      ((List<Object>) deserializedData).add(value);
    }
    unPkNumberObjects++;
    this.setUnPkCurrentObjectLength(-1);
  }

  public void addBuiltMessage(ChannelMessage channelMessage) {
    builtMessages.add(channelMessage);
  }

  public ChannelMessageReleaseCallback getReleaseListener() {
    return releaseListener;
  }

  public int getOriginatingId() {
    return originatingId;
  }

  public Queue<DataBuffer> getBuffers() {
    return buffers;
  }

  public ReceivedState getReceivedState() {
    return receivedState;
  }

  public void setReceivedState(ReceivedState receivedState) {
    this.receivedState = receivedState;
  }

  public Queue<ChannelMessage> getBuiltMessages() {
    return builtMessages;
  }

  public Object getDeserializedData() {
    return deserializedData;
  }

  public int getUnPkCurrentObjectLength() {
    return unPkCurrentObjectLength;
  }

  public void setUnPkCurrentObjectLength(int unPkCurrentObjectLength) {
    this.unPkCurrentObjectLength = unPkCurrentObjectLength;
  }

  public int getUnPkCurrentKeyLength() {
    return unPkCurrentKeyLength;
  }

  public void setUnPkCurrentKeyLength(int unPkCurrentKeyLength) {
    this.unPkCurrentKeyLength = unPkCurrentKeyLength;
  }

  public int getUnPkNumberObjects() {
    return unPkNumberObjects;
  }

  public int getUnPkBuffers() {
    return unPkBuffers;
  }

  public void incrementUnPkBuffers() {
    unPkBuffers++;
  }

  public boolean isKeyed() {
    return keyed;
  }

  public boolean isReadingKey() {
    return readingKey;
  }

  public void setReadingKey(boolean readingKey) {
    this.readingKey = readingKey;
  }

  public ObjectBuilderImpl getDataBuilder() {
    return dataBuilder;
  }

  public ObjectBuilderImpl getKeyBuilder() {
    return keyBuilder;
  }
}
