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

package org.apache.storm.spout;

import java.util.Collections;
import java.util.List;
import java.util.logging.Logger;

import org.apache.storm.topology.twister2.EdgeFieldMap;
import org.apache.storm.tuple.Fields;
import org.apache.storm.tuple.Twister2TupleWrapper;
import org.apache.storm.utils.Utils;

import edu.iu.dsc.tws.api.compute.TaskContext;

public class SpoutOutputCollector implements ISpoutOutputCollector {

  private static final Logger LOG = Logger.getLogger(SpoutOutputCollector.class.getName());

  private final TaskContext taskContext;
  private final String spoutId;
  private final EdgeFieldMap keyedOutEdges;
  private final EdgeFieldMap outFieldsForEdge;

  /**
   * Initializes a SpoutOutputCollector
   *
   * @param taskContext the instance of twister2 task context
   */
  public SpoutOutputCollector(String spoutId,
                              TaskContext taskContext,
                              EdgeFieldMap outFieldsForEdge,
                              EdgeFieldMap keyedOutEdges) {
    this.spoutId = spoutId;
    this.taskContext = taskContext;
    this.keyedOutEdges = keyedOutEdges;
    this.outFieldsForEdge = outFieldsForEdge;
  }

  /**
   * Emits a tuple to the default output stream with a null message id. Storm will
   * not track this message so ack and fail will never be called for this tuple. The
   * emitted values must be immutable.
   */
  public List<Integer> emit(List<Object> tuple) {
    return emit(tuple, null);
  }

  /**
   * Emits a new tuple to the default output stream with the given message ID.
   * When Storm detects that this tuple has been fully processed, or has failed
   * to be fully processed, the spout will receive an ack or fail callback respectively
   * with the messageId as long as the messageId was not null. If the messageId was null,
   * Storm will not track the tuple and no callback will be received.
   * Note that Storm's event logging functionality will only work if the messageId
   * is serializable via Kryo or the Serializable interface. The emitted values must be immutable.
   *
   * @return the list of task ids that this tuple was sent to
   */
  public List<Integer> emit(List<Object> tuple, Object messageId) {
    return emit(Utils.getDefaultStream(this.spoutId), tuple, messageId);
  }

  @Override
  public List<Integer> emit(String streamId, List<Object> tuple, Object messageId) {
    LOG.finest("Writing to the stream " + streamId + " data : " + tuple);
    //todo remove tupleWrapper once core level List handling issue is fixed
    Twister2TupleWrapper tupleWrapper = new Twister2TupleWrapper(tuple);
    if (!this.keyedOutEdges.containsKey(streamId)) {
      this.taskContext.write(streamId, tupleWrapper);
    } else {
      Fields allFields = outFieldsForEdge.get(streamId);
      Fields fieldsForKey = keyedOutEdges.get(streamId);
      List<Object> key = allFields.select(fieldsForKey, tuple);
      this.taskContext.write(streamId, key, tupleWrapper);
    }
    //todo return task ids, not yet supported by twister2
    return Collections.singletonList(0);
  }

  @Override
  public void emitDirect(int taskId, String streamId, List<Object> tuple, Object messageId) {

  }

  @Override
  public void reportError(Throwable error) {

  }
}
