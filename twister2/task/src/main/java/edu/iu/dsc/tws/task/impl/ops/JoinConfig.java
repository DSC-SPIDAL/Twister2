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
package edu.iu.dsc.tws.task.impl.ops;

import java.util.Comparator;

import edu.iu.dsc.tws.api.comms.CommunicationContext;
import edu.iu.dsc.tws.api.comms.messaging.types.MessageType;
import edu.iu.dsc.tws.api.comms.messaging.types.MessageTypes;
import edu.iu.dsc.tws.api.comms.packing.MessageSchema;
import edu.iu.dsc.tws.api.compute.OperationNames;
import edu.iu.dsc.tws.api.compute.graph.Edge;
import edu.iu.dsc.tws.task.impl.ComputeConnection;
import edu.iu.dsc.tws.task.impl.ComputeConnectionUtils;

public class JoinConfig extends AbstractKeyedOpsConfig<JoinConfig> {
  private Comparator keyCompartor;

  private String rightSource;

  private String rightEdgeName;

  private MessageType rightOpDataType = MessageTypes.OBJECT;

  private String group;

  private MessageSchema leftMessageSchema = MessageSchema.noSchema();
  private MessageSchema rightMessageSchema = MessageSchema.noSchema();

  public JoinConfig(String leftParent, String rightParent,
                    ComputeConnection computeConnection,
                    CommunicationContext.JoinType joinType,
                    CommunicationContext.JoinAlgorithm joinAlgorithm) {
    super(leftParent, OperationNames.JOIN, computeConnection);
    this.rightSource = rightParent;
    this.withProperty(CommunicationContext.USE_DISK, false);
    this.withProperty(CommunicationContext.JOIN_TYPE, joinType);
    this.withProperty(CommunicationContext.JOIN_ALGORITHM, joinAlgorithm);
  }

  public JoinConfig useDisk(boolean useDisk) {
    return this.withProperty(CommunicationContext.USE_DISK, useDisk);
  }

  public <T> JoinConfig withComparator(Comparator<T> keyComparator) {
    this.keyCompartor = keyComparator;
    return this.withProperty(CommunicationContext.KEY_COMPARATOR, keyComparator);
  }

  public JoinConfig viaLeftEdge(String edge) {
    this.edgeName = edge;
    return this;
  }

  public JoinConfig viaRightEdge(String edge) {
    this.rightEdgeName = edge;
    return this;
  }

  public JoinConfig withRightDataType(MessageType dataType) {
    this.rightOpDataType = dataType;
    return this;
  }

  public JoinConfig withLeftDataType(MessageType dataType) {
    this.opDataType = dataType;
    return this;
  }

  @Override
  public JoinConfig withMessageSchema(MessageSchema messageSchema) {
    this.leftMessageSchema = messageSchema;
    this.rightMessageSchema = messageSchema;
    return this;
  }

  public JoinConfig withLeftMessageSchema(MessageSchema messageSchema) {
    this.leftMessageSchema = messageSchema;
    return this;
  }

  public JoinConfig withRightMessageSchema(MessageSchema messageSchema) {
    this.rightMessageSchema = messageSchema;
    return this;
  }

  public JoinConfig withTargetEdge(String g) {
    this.group = g;
    return this;
  }

  @Override
  void validate() {
    if (this.keyCompartor == null) {
      failValidation("Join operation needs a key comparator.");
    }

    if (this.rightEdgeName == null) {
      failValidation("Right edge should have a name");
    }
  }

  public ComputeConnection connect() {
    Edge leftEdge = this.buildEdge();
    leftEdge.setEdgeIndex(0);
    leftEdge.setNumberOfEdges(2);
    leftEdge.setMessageSchema(this.leftMessageSchema);

    Edge rightEdge = this.buildRightEdge();
    rightEdge.setEdgeIndex(1);
    rightEdge.setNumberOfEdges(2);
    rightEdge.setMessageSchema(this.rightMessageSchema);

    // we generate the name
    if (group == null) {
      group = edgeName + "-" + rightEdgeName + "-" + source + "-" + rightSource;
    }
    leftEdge.setTargetEdge(group);
    rightEdge.setTargetEdge(group);

    ComputeConnectionUtils.connectEdge(this.computeConnection, this.source, leftEdge);
    ComputeConnectionUtils.connectEdge(this.computeConnection, this.rightSource, rightEdge);
    return this.computeConnection;
  }

  Edge buildRightEdge() {
    this.runValidation();
    Edge edge = new Edge(this.rightEdgeName, this.operationName);
    edge.setDataType(rightOpDataType);
    edge.addProperties(propertiesMap);
    updateEdge(edge);

    edge.setKeyed(true);
    edge.setKeyType(opKeyType);
    edge.setPartitioner(this.tPartitioner);
    return edge;
  }

  @Override
  protected Edge updateEdge(Edge newEdge) {
    return newEdge;
  }
}
