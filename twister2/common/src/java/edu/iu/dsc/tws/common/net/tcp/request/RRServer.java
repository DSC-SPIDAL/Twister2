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
package edu.iu.dsc.tws.common.net.tcp.request;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.channels.SocketChannel;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.logging.Level;
import java.util.logging.Logger;

import com.google.common.collect.HashBiMap;
import com.google.protobuf.InvalidProtocolBufferException;
import com.google.protobuf.Message;

import edu.iu.dsc.tws.api.config.Config;
import edu.iu.dsc.tws.api.net.StatusCode;
import edu.iu.dsc.tws.api.net.request.ConnectHandler;
import edu.iu.dsc.tws.api.net.request.MessageHandler;
import edu.iu.dsc.tws.api.net.request.RequestID;
import edu.iu.dsc.tws.common.net.tcp.ChannelHandler;
import edu.iu.dsc.tws.common.net.tcp.Progress;
import edu.iu.dsc.tws.common.net.tcp.Server;
import edu.iu.dsc.tws.common.net.tcp.TCPMessage;

/**
 * RRServer class is used by Job Master
 * It works in request/response messaging
 *
 * Workers and the client always send a request message, and
 * JobMaster sends a single response message to each request message
 *
 * However, sometimes job master may send messages to workers that are not response messages
 * For example, the driver in Job Master may send messages to workers
 * The messages that are initiated from Job Master are one-way messages
 * They don't have response messages
 *
 * Message Format:
 * RequestID (32 bytes), message type length, message type data, senderID (4 bytes), message data
 * message type is the class name of the protocol buffer for that message
 *
 * RequestID is generated in request senders (workers or the client),
 * and the same requestID is used in the response message.
 *
 * When job master sends a message that is not a response to a request,
 * it uses the DUMMY_REQUEST_ID as the requestID.
 */
public class RRServer {
  private static final Logger LOG = Logger.getLogger(RRServer.class.getName());

  protected Server server;

  /**
   * We keep track of connected channels here to make sure we close them
   */
  protected List<SocketChannel> connectedChannels = new ArrayList<>();

  /**
   * worker channels with workerIDs
   */
  protected HashBiMap<SocketChannel, Integer> workerChannels = HashBiMap.create();

  /**
   * the client channel,
   */
  protected SocketChannel clientChannel;

  /**
   * Keep track of the request handler using protocol buffer message types
   */
  protected Map<String, MessageHandler> requestHandlers = new HashMap<>();

  /**
   * Message type name to builder
   */
  protected Map<String, Message.Builder> messageBuilders = new HashMap<>();

  /**
   * Keep track of the requests
   */
  protected Map<RequestID, SocketChannel> requestChannels = new HashMap<>();

  /**
   * Job Master ID
   */
  protected int serverID;

  /**
   * The client id
   */
  public static final int CLIENT_ID = -100;

  /**
   * Connection handler
   */
  protected ConnectHandler connectHandler;

  /**
   * The loop that executes the selector
   */
  protected Progress loop;

  /**
   * We keep track of pending send count to determine weather all the sends are completed
   */
  protected int pendingSendCount = 0;

  public RRServer(Config cfg, String host, int port, Progress looper, int serverID,
                  ConnectHandler cHandler) {
    this.connectHandler = cHandler;
    this.loop = looper;
    this.serverID = serverID;
    server = new Server(cfg, host, port, loop, new Handler(), false);
  }

  public void registerRequestHandler(Message.Builder builder, MessageHandler handler) {
    requestHandlers.put(builder.getDescriptorForType().getFullName(), handler);
    messageBuilders.put(builder.getDescriptorForType().getFullName(), builder);
  }

  public void start() {
    server.start();
  }

  public void stop() {
    server.stop();
  }

  public void stopGraceFully(long waitTime) {
    // now lets wait if there are messages pending
    long start = System.currentTimeMillis();

    boolean pending;
    long elapsed;
    do {
      loop.loop();
      pending = server.hasPending();
      elapsed = System.currentTimeMillis() - start;
    } while ((pending || pendingSendCount != 0 || connectedChannels.size() > 0)
        && elapsed < waitTime);

    // after sometime we need to stop
    stop();
  }

  public Set<Integer> getConnectedWorkers() {
    return workerChannels.values();
  }

  /**
   * Send a response to a request id
   * @param requestID request id
   * @param message message
   * @return true if response was accepted
   */
  public boolean sendResponse(RequestID requestID, Message message) {

    if (!requestChannels.containsKey(requestID)) {
      LOG.log(Level.SEVERE, "Trying to send a response to non-existing request");
      return false;
    }

    SocketChannel channel = requestChannels.get(requestID);

    if (channel == null) {
      LOG.log(Level.SEVERE, "Channel is NULL for response");
    }

    if (!workerChannels.containsKey(channel) && !channel.equals(clientChannel)) {
      LOG.log(Level.WARNING, "Failed to send response on disconnected socket");
      return false;
    }

    TCPMessage tcpMessage = sendMessage(message, requestID, channel);

    if (tcpMessage != null) {
      requestChannels.remove(requestID);
      return true;
    } else {
      return false;
    }
  }

  /**
   * Send a non-response message to a worker or to the client
   * @param message message
   * @return true if response was accepted
   */
  public boolean sendMessage(Message message, int targetID) {

    SocketChannel channel;
    if (targetID == CLIENT_ID) {
      if (clientChannel == null) {
        LOG.severe("Trying to send a message to the client, but it has not connected yet.");
        return false;
      }
      channel = clientChannel;
    } else if (workerChannels.containsValue(targetID)) {
      channel = workerChannels.inverse().get(targetID);
    } else {
      LOG.severe("Trying to send a message to a worker that has not connected yet. workerID: "
          + targetID);
      return false;
    }

    // this is most likely not needed, but just to make sure
    if (channel == null) {
      LOG.log(Level.SEVERE, "Channel is NULL for response");
      return false;
    }

    // since this is not a request/response message, we put the dummy request id
    RequestID dummyRequestID = RequestID.DUMMY_REQUEST_ID;

    TCPMessage tcpMessage = sendMessage(message, dummyRequestID, channel);

    return tcpMessage != null;
  }

  protected TCPMessage sendMessage(Message message, RequestID requestID, SocketChannel channel) {
    byte[] data = message.toByteArray();
    String messageType = message.getDescriptorForType().getFullName();

    // lets serialize the message
    int capacity = requestID.getId().length + data.length + messageType.getBytes().length + 8;
    ByteBuffer buffer = ByteBuffer.allocate(capacity);
    // we send message id, worker id and data
    buffer.put(requestID.getId());
    // pack the name of the message
    ByteUtils.packString(messageType, buffer);
    // pack the worker id
    buffer.putInt(serverID);
    // pack data
    buffer.put(data);

    TCPMessage send = server.send(channel, buffer, capacity, 0);
    if (send != null) {
      pendingSendCount++;
    }
    return send;
  }

  private class Handler implements ChannelHandler {
    @Override
    public void onError(SocketChannel channel) {
      workerChannels.remove(channel);
      connectedChannels.remove(channel);
      connectHandler.onError(channel);

      loop.removeAllInterest(channel);

      try {
        channel.close();
        LOG.log(Level.FINEST, "Closed the channel: " + channel);
      } catch (IOException e) {
        LOG.log(Level.SEVERE, "Channel closed error: " + channel, e);
      }
    }

    @Override
    public void onConnect(SocketChannel channel, StatusCode status) {
      connectedChannels.add(channel);
      connectHandler.onConnect(channel, status);
    }

    @Override
    public void onClose(SocketChannel channel) {
      workerChannels.remove(channel);
      connectedChannels.remove(channel);
      connectHandler.onClose(channel);

      if (channel.equals(clientChannel)) {
        clientChannel = null;
      }
    }

    @Override
    public void onReceiveComplete(SocketChannel channel, TCPMessage readRequest) {
      if (channel == null) {
        LOG.log(Level.SEVERE, "Chanel on receive is NULL");
      }

      // read headers amd the message
      ByteBuffer data = readRequest.getByteBuffer();

      // read requestID
      byte[] requestIDBytes = new byte[RequestID.ID_SIZE];
      data.get(requestIDBytes);
      RequestID requestID = RequestID.fromBytes(requestIDBytes);

      // unpack the string
      String messageType = ByteUtils.unPackString(data);

      // now get sender worker id
      int senderID = data.getInt();

      Message.Builder builder = messageBuilders.get(messageType);
      if (builder == null) {
        throw new RuntimeException("Received response without a registered response");
      }

      try {
        builder.clear();

        // size of the header
        int headerLength = 8 + requestIDBytes.length + messageType.getBytes().length;
        int dataLength = readRequest.getLength() - headerLength;

        // reconstruct protocol buffer message
        byte[] d = new byte[dataLength];
        data.get(d);
        builder.mergeFrom(d);
        Message message = builder.build();

        // save this channel
        saveChannel(channel, senderID, message);

        LOG.log(Level.FINEST, String.format("Adding channel %s", new String(requestIDBytes)));
        requestChannels.put(requestID, channel);

        MessageHandler handler = requestHandlers.get(messageType);
        handler.onMessage(requestID, senderID, message);
      } catch (InvalidProtocolBufferException e) {
        LOG.log(Level.SEVERE, "Failed to build a message", e);
      }
    }

    @Override
    public void onSendComplete(SocketChannel channel, TCPMessage writeRequest) {
      pendingSendCount--;
    }
  }

  /**
   * remove the channel when the worker is removed
   * @param workerID
   */
  public void removeWorkerChannel(int workerID) {
    SocketChannel removedChannel = workerChannels.inverse().remove(workerID);
    if (removedChannel == null) {
      return;
    }

    try {
      removedChannel.close();
    } catch (IOException e) {
      LOG.log(Level.WARNING, "Exception when closing the channel: ", e);
    }
  }

  /**
   * save if it is a new channel
   * if it is a client channel, save it in that variable
   * if it is a new worker channel that will get its id from the job master
   * save it in the temporary variable
   * otherwise add it to the channel list
   */
  private void saveChannel(SocketChannel channel, int senderID, Message message) {

    // if the channel already exist, do nothing
    if (workerChannels.containsKey(channel)) {
      return;
    }

    // if it is the client
    // set it, no need to check whether it is already set
    // since it does not harm setting again
    if (senderID == CLIENT_ID) {
      clientChannel = channel;
      LOG.info("Message received from submitting client. Channel set.");
      return;
    }

    // if there is already a channel for this worker,
    // replace it with this one
    if (workerChannels.inverse().containsKey(senderID)) {
      LOG.warning(String.format("While there is a channel for workerID[%d], "
          + "another channel connected from the same worker. Replacing older one. ", senderID));
    }

    workerChannels.forcePut(channel, senderID);
  }
}
