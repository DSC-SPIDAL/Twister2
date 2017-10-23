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
package edu.iu.dsc.tws.comms.mpi;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.logging.Logger;

import edu.iu.dsc.tws.comms.api.MessageHeader;
import edu.iu.dsc.tws.comms.routing.BinaryTreeRouter;
import edu.iu.dsc.tws.comms.routing.IRouter;
import edu.iu.dsc.tws.comms.routing.Routing;

public class MPIDataFlowBroadcast extends MPIDataFlowOperation {
  private static final Logger LOG = Logger.getLogger(MPIDataFlowBroadcast.class.getName());

  private int source;

  private Set<Integer> destinations;

  public MPIDataFlowBroadcast(TWSMPIChannel channel, int src, Set<Integer> dests) {
    super(channel);
    this.source = src;
    this.destinations = dests;
  }

  @Override
  protected void sendCompleteMPIMessage(int src, MPIMessage mpiMessage) {
    List<Integer> routes = new ArrayList<>();
    routeSendMessage(src, mpiMessage.getHeader(), routes);
    if (routes.size() == 0) {
      throw new RuntimeException("Failed to get downstream tasks");
    }
    sendMessage(mpiMessage, routes);
  }

  @Override
  public void close() {
  }

  @Override
  public void injectPartialResult(int src, Object message) {
    throw new RuntimeException("Not supported method");
  }

  protected void passMessageDownstream(MPIMessage currentMessage) {
    List<Integer> routes = new ArrayList<>();
    // we will get the routing based on the originating id
    routeReceivedMessage(currentMessage.getHeader(), routes);
    // try to send further
    sendMessage(currentMessage, routes);
  }

  protected IRouter setupRouting() {
    // lets create the routing needed
    BinaryTreeRouter tree = new BinaryTreeRouter();
    // we will only have one distinct route
    Set<Integer> sources = new HashSet<>();
    sources.add(source);

    tree.init(config, instancePlan, sources, destinations, edge, 1);
    return tree;
  }

  @Override
  protected void routeReceivedMessage(MessageHeader message, List<Integer> routes) {
    // check the origin
    int src = message.getSourceId();
    // get the expected routes
    Routing routing = expectedRoutes.get(src);

    if (routing == null) {
      throw new RuntimeException("Un-expected message from source: " + src);
    }

    routes.addAll(routing.getDownstreamIds());
  }

  @Override
  protected void routeSendMessage(int src, MessageHeader message, List<Integer> routes) {
    // get the expected routes
    Routing routing = expectedRoutes.get(src);

    if (routing == null) {
      throw new RuntimeException("Un-expected message from source: " + src);
    }

    routes.addAll(routing.getDownstreamIds());
  }
}

