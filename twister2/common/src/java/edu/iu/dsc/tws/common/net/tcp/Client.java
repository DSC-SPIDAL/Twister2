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
package edu.iu.dsc.tws.common.net.tcp;

import java.io.IOException;
import java.net.InetSocketAddress;
import java.nio.ByteBuffer;
import java.nio.channels.SelectableChannel;
import java.nio.channels.SocketChannel;
import java.util.logging.Level;
import java.util.logging.Logger;

import edu.iu.dsc.tws.api.config.Config;
import edu.iu.dsc.tws.api.net.StatusCode;

public class Client implements SelectHandler {
  private static final Logger LOG = Logger.getLogger(Client.class.getName());

  /**
   * The socket channel
   */
  private SocketChannel socketChannel;

  /**
   * Network address
   */
  private InetSocketAddress address;

  /**
   * Configuration
   */
  private Config config;

  /**
   * Selector
   */
  private Progress progress;

  /**
   * The channel to read and receive
   */
  private BaseNetworkChannel channel;

  /**
   * Weather we are connected
   */
  private boolean isConnected;

  private boolean connecting = false;

  private boolean tryToConnect = false;

  /**
   * The channel callback
   */
  private ChannelHandler channelHandler;

  /**
   * Fixed buffers
   */
  private boolean fixedBuffers = true;

  public Client(String host, int port, Config cfg, Progress looper, ChannelHandler handler) {
    address = new InetSocketAddress(host, port);
    config = cfg;
    isConnected = false;
    progress = looper;
    channelHandler = handler;
  }

  public Client(String host, int port, Config cfg, Progress looper,
                ChannelHandler handler, boolean fixBuffers) {
    address = new InetSocketAddress(host, port);
    config = cfg;
    isConnected = false;
    progress = looper;
    channelHandler = handler;
    fixedBuffers = fixBuffers;
  }

  /**
   * this method must be called when the client is disconnected
   */
  public void setHostAndPort(String host, int port) {
    address = new InetSocketAddress(host, port);
  }

  public boolean connect() {
    try {
      socketChannel = SocketChannel.open();
      socketChannel.configureBlocking(false);
      socketChannel.socket().setTcpNoDelay(true);

      if (!tryToConnect) {
        LOG.log(Level.INFO, String.format("Connecting to the server on %s:%d",
            address.getHostName(), address.getPort()));
      }

      if (socketChannel.connect(address)) {
        handleConnect(socketChannel);
      } else {
        connecting = true;
        progress.registerConnect(socketChannel, this);
      }
    } catch (IOException e) {
      if (tryToConnect) {
        tryToConnect = false;
        return false;
      }
      LOG.log(Level.SEVERE, "Error connecting to remote endpoint: " + address, e);
      channelHandler.onError(socketChannel, StatusCode.ERROR_CONN);
      return false;
    }

    return true;
  }

  /**
   * this method may be called when the target machine may not be up yet
   * this method may be called repeatedly, until it connects
   * connection exceptions are ignored, they are not propagated to connection handler
   * connection attempt fails silently
   */
  public void tryConnecting() {
    if (connecting) {
      return;
    }

    tryToConnect = true;

    connect();
  }

  public boolean isConnected() {
    return isConnected;
  }

  public TCPMessage send(SocketChannel sc, ByteBuffer buffer, int size, int edge) {
    if (sc != socketChannel) {
      return null;
    }

    if (!isConnected) {
      return null;
    }

    // we set the limit to size and position to 0 in-order to write this message
    buffer.limit(size);
    buffer.position(0);

    channel.enableWriting();
    TCPMessage request = new TCPMessage(buffer.duplicate(), edge, size);
    if (channel.addWriteRequest(request)) {
      return request;
    }
    return null;
  }

  public TCPMessage receive(SocketChannel sc, ByteBuffer buffer, int size, int edge) {
    if (sc != socketChannel) {
      return null;
    }

    if (!isConnected) {
      return null;
    }

    TCPMessage request = new TCPMessage(buffer, edge, size);
    channel.addReadRequest(request);

    return request;
  }

  public void disconnect() {
    if (isConnected || connecting) {
      channel.forceFlush();
      progress.removeAllInterest(socketChannel);

      try {
        socketChannel.close();
        // we call the onclose with null value
        channelHandler.onClose(socketChannel);
        isConnected = false;
        connecting = false;
        tryToConnect = false;
      } catch (IOException e) {
        LOG.log(Level.SEVERE, "Failed to stop Client", e);
      }
    }
  }

  /**
   * Stop the client while trying to process any queued requests and responses
   */
  public void disconnectGraceFully(long waitTime) {
    // now lets wait if there are messages pending
    long start = System.currentTimeMillis();

    boolean pending;
    long elapsed;
    do {
      pending = false;
      if (channel.isPending()) {
        progress.loop();
        pending = true;
      }
      elapsed = System.currentTimeMillis() - start;
    } while (pending && elapsed < waitTime);

    // after sometime we need to stop
    disconnect();
  }

  @Override
  public void handleRead(SelectableChannel ch) {
    channel.read();
  }

  @Override
  public void handleWrite(SelectableChannel ch) {
    channel.write();
  }

  @Override
  public void handleAccept(SelectableChannel ch) {
    throw new RuntimeException("Client cannot accept connections");
  }

  @Override
  public void handleConnect(SelectableChannel selectableChannel) {

    try {
      if (socketChannel.finishConnect()) {
        progress.unregisterConnect(selectableChannel);
        if (fixedBuffers) {
          channel = new FixedBufferChannel(config, progress, this, socketChannel, channelHandler);
        } else {
          channel = new DynamicBufferChannel(config, progress, this, socketChannel, channelHandler);
        }
        channel.enableReading();
        channel.enableWriting();

        isConnected = true;
        connecting = false;
        tryToConnect = false;
        channelHandler.onConnect(socketChannel);
      } else {
        if (!tryToConnect) {
          LOG.log(Level.SEVERE, "Failed to FinishConnect to endpoint: " + address);
          channelHandler.onError(socketChannel, StatusCode.ERROR_CONN);
        }
      }
    } catch (java.net.ConnectException e) {
      if (!tryToConnect) {
        channelHandler.onError(socketChannel, StatusCode.ERROR_CONN);
      }
    } catch (Exception e) {
      if (!tryToConnect) {
        LOG.log(Level.SEVERE, "Failed to FinishConnect to endpoint: " + address, e);
        channelHandler.onError(socketChannel, StatusCode.ERROR_CONN);
      }
    }
    connecting = false;
    tryToConnect = false;
  }

  @Override
  public void handleError(SelectableChannel ch) {
    channel.clear();
    progress.removeAllInterest(ch);

    LOG.log(Level.SEVERE, "Error on channel " + ch);

    try {
      ch.close();
    } catch (IOException e) {
      LOG.log(Level.SEVERE, "Failed to close connection in handleError", e);
    }

    isConnected = false;
    connecting = false;
    tryToConnect = false;
    channelHandler.onError(socketChannel, StatusCode.ERROR_CONN);
  }

  SocketChannel getSocketChannel() {
    return socketChannel;
  }
}
