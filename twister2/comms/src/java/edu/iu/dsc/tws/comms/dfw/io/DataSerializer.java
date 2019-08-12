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
package edu.iu.dsc.tws.comms.dfw.io;

import java.nio.ByteBuffer;

import edu.iu.dsc.tws.api.comms.packing.DataBuffer;
import edu.iu.dsc.tws.api.comms.packing.DataPacker;
import edu.iu.dsc.tws.comms.dfw.OutMessage;

/**
 * Builds the message and copies it into data buffers.The structure of the message depends on the
 * type of message that is sent, for example if it is a keyed message or not.
 * <p>
 * The main structure of the built message is |Header|Body|.
 * <p>
 * The header has the following structure
 * |source|flags|destinationID|numberOfMessages|,
 * source - source of the message
 * flags - message flags
 * destinationId - where the message is sent
 * numberOfMessages - number of messages
 * <p>
 * Header can be followed by 0 or more messages, each message will have the following structure
 * |length(integer)|message body|
 * <p>
 * For a keyed message the message body consists of two parts
 * |key|body|
 * <p>
 * For some keys we need to send the length of the key, i.e. byte arrays and objects. In that case
 * key consists of two parts
 * |key length(integer)|actual key|
 * <p>
 * For other cases such as integer or double keys, we know the length of the key, so we only send
 * the key.
 */
public class DataSerializer extends BaseSerializer {

  /**
   * Builds the body of the message. Based on the message type different build methods are called
   *
   * @param payload the message that needs to be built
   * @param sendMessage the send message object that contains all the metadata
   * @param targetBuffer the data targetBuffer to which the built message needs to be copied
   * @return true if the body was built and copied to the targetBuffer successfully,false otherwise.
   */
  public boolean serializeSingleMessage(Object payload,
                                        OutMessage sendMessage, DataBuffer targetBuffer) {
    return serializeData(payload, sendMessage.getSerializationState(),
        targetBuffer, sendMessage.getDataType().getDataPacker());
  }

  /**
   * Helper method that builds the body of the message for regular messages.
   *
   * @param payload the message that needs to be built
   * @param state the state object of the message
   * @param targetBuffer the data targetBuffer to which the built message needs to be copied
   * @return true if the body was built and copied to the targetBuffer successfully,false otherwise.
   */
  protected boolean serializeData(Object payload, SerializeState state,
                                DataBuffer targetBuffer, DataPacker dataPacker) {
    ByteBuffer byteBuffer = targetBuffer.getByteBuffer();
    // okay we need to serialize the header
    if (state.getPart() == SerializeState.Part.INIT) {
      // okay we need to serialize the data
      int dataLength = dataPacker.determineLength(payload, state);
      state.getActive().setTotalToCopy(dataLength);

      state.setCurrentHeaderLength(dataLength);

      // add the header bytes to the total bytes
      state.setPart(SerializeState.Part.HEADER);
    }

    if (state.getPart() == SerializeState.Part.HEADER) {
      // first we need to copy the data size to buffer
      if (buildSubMessageHeader(targetBuffer, state.getCurrentHeaderLength())) {
        return false;
      }
      state.setPart(SerializeState.Part.BODY);
    }

    // now we can serialize the body
    if (state.getPart() != SerializeState.Part.BODY) {
      // now set the size of the buffer
      targetBuffer.setSize(byteBuffer.position());
      return false;
    }

    boolean completed = DataPackerProxy.writeDataToBuffer(
        dataPacker,
        payload,
        byteBuffer,
        state
    );

    // now set the size of the buffer
    targetBuffer.setSize(byteBuffer.position());

    // okay we are done with the message
    return state.reset(completed);
  }

  /**
   * Builds the sub message header which is used in multi messages to identify the lengths of each
   * sub message. The structure of the sub message header is |length + (key length)|. The key length
   * is added for keyed messages
   */
  private boolean buildSubMessageHeader(DataBuffer buffer, int length) {
    ByteBuffer byteBuffer = buffer.getByteBuffer();
    if (byteBuffer.remaining() < NORMAL_SUB_MESSAGE_HEADER_SIZE) {
      return true;
    }
    byteBuffer.putInt(length);
    return false;
  }
}
